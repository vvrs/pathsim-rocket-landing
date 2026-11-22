"""Rocket landing as one convex problem, re-solved every second (MPC).

Plain functions only, so the same code drives the PathSim custom blocks
(rocket_blocks.py) and the code panel of the PathView models (make_pvm.py
pastes this file in verbatim).

Problem: CVXPY x NASA short course 2024, Lecture 3 (point mass, side view):
    minimize    sum_k ||f_k||                       (fuel)
    subject to  v[k+1] = v[k] + (h/m) f[k] - h g e_z
                p[k+1] = p[k] + (h/2) (v[k] + v[k+1])
                ||f_k|| <= Fmax                     (thrust ceiling)
                z_k >= tan(alpha) |x_k|             (glide cone)
                p_K = 0, v_K = 0                    (touch down at rest on the pad)
"""
import time
import numpy as np
import cvxpy as cp

# ---- lecture values (CVXPY x NASA Lecture 3) -----------------------------------
H, G, M, FMAX, ALPHA, K = 1.0, 0.1, 10.0, 10.0, np.pi / 8, 35
P0, V0 = (80.0, 100.0), (-12.0, -6.0)          # start: 80 m from the pad, 100 m up, heading toward it

# wind gusts [m/s^2]: new value every 3 s, calm after t = 26 s (x, z)
GUST_TABLE = [(-0.133715, -0.000130), (-0.133715, -0.000130), (-0.133715, -0.000130), (0.036539, -0.084836), (0.036539, -0.084836), (0.036539, -0.084836), (-0.126747, 0.077078), (-0.126747, 0.077078), (-0.126747, 0.077078), (-0.154649, -0.066641), (-0.154649, -0.066641), (-0.154649, -0.066641), (0.161398, 0.021939), (0.161398, 0.021939), (0.161398, 0.021939), (-0.047162, 0.002050), (-0.047162, 0.002050), (-0.047162, 0.002050), (0.058623, -0.040444), (0.058623, -0.040444), (0.058623, -0.040444), (-0.130331, 0.051847), (-0.130331, 0.051847), (-0.130331, 0.051847), (0.061330, 0.002229), (0.061330, 0.002229), (0.000000, 0.000000), (0.000000, 0.000000), (0.000000, 0.000000), (0.000000, 0.000000), (0.000000, 0.000000), (0.000000, 0.000000), (0.000000, 0.000000), (0.000000, 0.000000), (0.000000, 0.000000)]

# ---- rigid body (illustrative values, not from the lecture) --------------------------
J_BODY = 30.0                    # pitch inertia [kg m^2]
ARM = 4.0                        # engine gimbal point below the centre of mass [m]
DELTA_MAX = np.deg2rad(0.0)      # gimbal limit [rad]: 0 = gimbal off; try 5-10 deg (its side force fights the plan)
TAU_RCS = 6.0                    # reaction-control torque limit [N m]: turns the body, burning or coasting
K_ANG, K_RATE = 1.0, 60.0        # attitude loop: angle error -> rate command [1/s], rate error -> torque [N m s/rad]
OMEGA_MAX = 0.25                 # turn-rate limit [rad/s], so the body can always stop turning in time
FMAX_PLAN_RIGID = 9.0            # plan at 90 % thrust: keep margin for steering
TILT_MAX = np.deg2rad(80.0)      # rigid body: thrust within 80 deg of vertical (never push down)
TILT_LAND = np.deg2rad(15.0)     # ...and within 15 deg for the last 2 s, so it lands upright
SLEW_RIGID = 3.0                 # thrust vector may change by at most 3 N per second
MAX_DELAY = 8                    # touchdown may slip up to 8 s past t = 35 s


# ---- the convex plan ------------------------------------------------------------------
def min_fuel_plan(p0, v0, steps, fmax=FMAX, tilt_max=None, tilt_land=None, slew=None, f_prev=None):
    """Solve the landing SOCP from (p0, v0) over `steps` one-second steps.
    Returns (P, V, F, fuel, status). If the hard landing is infeasible (possible
    when a real body drifts off the plan), it lands as close as it can instead.
    tilt_max / tilt_land add the thrust-pointing cone ||f_k|| <= f_z,k / cos(tilt): convex too.
    slew limits how much the thrust vector may change per step, ||f_k+1 - f_k|| <= slew,
    starting from the thrust applied last (f_prev): also convex."""
    P = cp.Variable((steps + 1, 2)); V = cp.Variable((steps + 1, 2)); F = cp.Variable((steps, 2))
    dyn = [P[0] == np.asarray(p0, float), V[0] == np.asarray(v0, float),
           V[1:, 0] == V[:-1, 0] + H * F[:, 0] / M,
           V[1:, 1] == V[:-1, 1] + H * (F[:, 1] / M - G),
           P[1:] == P[:-1] + H / 2 * (V[:-1] + V[1:]),
           cp.norm(F, 2, axis=1) <= fmax,
           P[:, 1] >= np.tan(ALPHA) * cp.abs(P[:, 0])]
    if tilt_max is not None:
        dyn += [cp.norm(F, 2, axis=1) <= F[:, 1] / np.cos(tilt_max)]
    if slew is not None:
        dyn += [cp.norm(F[1:] - F[:-1], 2, axis=1) <= slew]
        if f_prev is not None:
            dyn += [cp.norm(F[0] - np.asarray(f_prev, float)) <= slew]
    if tilt_land is not None:
        last = min(2, steps)
        dyn += [cp.norm(F[-last:], 2, axis=1) <= F[-last:, 1] / np.cos(tilt_land)]
    fuel = cp.sum(cp.norm(F, 2, axis=1))
    prob = cp.Problem(cp.Minimize(fuel), dyn + [P[steps] == 0, V[steps] == 0])
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        soft = cp.Problem(cp.Minimize(fuel + 100 * cp.norm(P[steps]) + 100 * cp.norm(V[steps])),
                          dyn[:6] + dyn[7:])        # keep everything but the glide cone
        soft.solve(solver=cp.CLARABEL)
        return P.value, V.value, F.value, float(fuel.value), "soft"
    return P.value, V.value, F.value, float(prob.value), prob.status


BURN_MIN = 1.0                   # thrust below 1 N (10 %) counts as "not burning" for attitude


def burn_direction(F, threshold=BURN_MIN):
    """Tilt [rad, + = nose toward +x] of the first real burn in a plan (0 = upright)."""
    for f in F:
        if np.hypot(*f) > threshold:
            return float(np.arctan2(f[0], f[1]))
    return 0.0


class ConvexLandingPlanner:
    """Called once per second with the measured state (x, z, vx, vz).

    mode="mpc":       re-solve from the measured state with a shrinking horizon,
                      return only the first thrust (model predictive control)
    mode="open_loop": solve once at t = 0, then play that thrust schedule blind

    Returns (fx, fz, theta_ref): planned thrust and the direction of the next burn,
    so a real body can turn toward it during a coast.

    If landing on time is no longer possible, the touchdown time moves later one
    second at a time, up to `max_delay` (a line search on the landing time).
    """

    def __init__(self, mode="mpc", steps=K, fmax=FMAX, tilt_max=None, tilt_land=None, slew=None, max_delay=0):
        self.mode, self.steps, self.fmax, self.max_delay = mode, steps, fmax, max_delay
        self.limits = dict(tilt_max=tilt_max, tilt_land=tilt_land, slew=slew)
        self.f_prev = None
        self.k, self.log, self._F = 0, [], None

    def __call__(self, x, z, vx, vz):
        k = self.k; self.k += 1
        left = self.steps - k
        if left <= 0:
            return np.array([0.0, 0.0, 0.0])
        t0 = time.perf_counter()
        if self.mode == "mpc" or self._F is None:
            for extra in range(self.max_delay + 1):
                P, V, F, fuel, status = min_fuel_plan((x, z), (vx, vz), left + extra, self.fmax,
                                                      f_prev=self.f_prev, **self.limits)
                if status != "soft":
                    break
            self.steps += extra; left += extra              # touchdown moved later by `extra` seconds
            self._F, self._P = F, P
            ms = 1e3 * (time.perf_counter() - t0)
            self.log.append(dict(k=k, state=[x, z, vx, vz], plan=P.tolist(), f0=F[0].tolist(), t_land=self.steps,
                                 fuel_to_go=fuel, status=status, solve_ms=ms))
            f, rest = F[0], F
        else:
            f, rest = self._F[k], self._F[k:]
        self.f_prev = np.array(f[:2], float)
        return np.array([f[0], f[1], burn_direction(rest)])


def gust(t, on=True):
    """Gust acceleration (dx, dz) at time t: piecewise constant, one value per second."""
    k = int(np.floor(t + 1e-9))
    if not on or k < 0 or k >= len(GUST_TABLE):
        return 0.0, 0.0
    return GUST_TABLE[k]


# ---- plants ------------------------------------------------------------------------------
def point_mass_rhs(s, u, t):
    """s = (x, z, vx, vz), u = (fx, fz, dx, dz)."""
    fx, fz, dx, dz = u
    return np.array([s[2], s[3], fx / M + dx, fz / M - G + dz])


def rigid_body_rhs(s, u, t):
    """Planar rigid body. s = (x, z, theta, vx, vz, omega), u = (T, delta, tau_rcs, dx, dz).
    Thrust T acts along theta + delta; the gimbal sits ARM below the centre of mass."""
    T, delta, tau, dx, dz = u
    th = s[2]
    return np.array([s[3], s[4], s[5],
                     T / M * np.sin(th + delta) + dx,
                     T / M * np.cos(th + delta) - G + dz,
                     (-T * ARM * np.sin(delta) + tau) / J_BODY])


def attitude_control(fx, fz, th_ref, th, om):
    """Inner loop: turn the body toward the planned thrust, then push along the body.
    Returns (T, delta, tau_rcs)."""
    fmag = np.hypot(fx, fz)
    th_des = np.arctan2(fx, fz) if fmag > BURN_MIN else th_ref
    T = float(np.clip(fx * np.sin(th) + fz * np.cos(th), 0.0, FMAX))   # only the part along the body helps
    err = (th_des - th + np.pi) % (2 * np.pi) - np.pi
    om_cmd = np.clip(K_ANG * err, -OMEGA_MAX, OMEGA_MAX)                 # turn at a capped rate...
    tau = K_RATE * (om_cmd - om)                                         # ...and track that rate
    delta = 0.0
    if T > 0.5:
        delta = float(np.arcsin(np.clip(-tau / (T * ARM), -np.sin(DELTA_MAX), np.sin(DELTA_MAX))))
    tau_rcs = float(np.clip(tau + T * ARM * np.sin(delta), -TAU_RCS, TAU_RCS))   # the rest from RCS
    return T, delta, tau_rcs
