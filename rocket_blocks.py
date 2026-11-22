"""Custom PathSim blocks for the rocket landing, with named ports.

    PointMassRocket   ODE      in: fx fz dx dz            out: x z vx vz
    RigidRocket       ODE      in: T delta tau_rcs dx dz  out: x z theta vx vz omega
    LandingMPC        Wrapper  in: x z vx vz              out: fx fz theta_ref   (runs every 1 s, holds its output)
    AttitudeLoop      Function in: fx fz theta_ref theta omega   out: T delta tau_rcs
    GustTable         Source   in: -                      out: dx dz
"""
import numpy as np
from pathsim.blocks import Block, ODE, Wrapper, Function

import rocket_core as rc


class PointMassRocket(ODE):
    """Point-mass rocket in the vertical plane (the lecture's model)."""
    input_port_labels = {"fx": 0, "fz": 1, "dx": 2, "dz": 3}
    output_port_labels = {"x": 0, "z": 1, "vx": 2, "vz": 3}

    def __init__(self, p0=rc.P0, v0=rc.V0):
        super().__init__(func=rc.point_mass_rhs, initial_value=np.array([*p0, *v0], float))


class RigidRocket(ODE):
    """Planar rigid body with a gimbaled engine and reaction-control torque."""
    input_port_labels = {"T": 0, "delta": 1, "tau_rcs": 2, "dx": 3, "dz": 4}
    output_port_labels = {"x": 0, "z": 1, "theta": 2, "vx": 3, "vz": 4, "omega": 5}

    def __init__(self, p0=rc.P0, v0=rc.V0, theta0=0.0, omega0=0.0):
        super().__init__(func=rc.rigid_body_rhs,
                         initial_value=np.array([p0[0], p0[1], theta0, v0[0], v0[1], omega0], float))


class LandingMPC(Wrapper):
    """Convex landing planner, sampled every T seconds with zero-order hold.
    mode="mpc" re-solves each sample; mode="open_loop" flies the t = 0 plan blind."""
    input_port_labels = {"x": 0, "z": 1, "vx": 2, "vz": 3}
    output_port_labels = {"fx": 0, "fz": 1, "theta_ref": 2}

    def __init__(self, mode="mpc", fmax=rc.FMAX, tilt_max=None, tilt_land=None, slew=None, max_delay=0, T=rc.H):
        self.planner = rc.ConvexLandingPlanner(mode=mode, fmax=fmax, tilt_max=tilt_max, tilt_land=tilt_land,
                                               slew=slew, max_delay=max_delay)
        super().__init__(func=self.planner, T=T)


class AttitudeLoop(Function):
    """Turns the body toward the planned thrust (gimbal while burning, RCS while coasting)."""
    input_port_labels = {"fx": 0, "fz": 1, "theta_ref": 2, "theta": 3, "omega": 4}
    output_port_labels = {"T": 0, "delta": 1, "tau_rcs": 2}

    def __init__(self):
        super().__init__(func=lambda fx, fz, th_ref, th, om: np.array(rc.attitude_control(fx, fz, th_ref, th, om)))


class GustTable(Block):
    """Piecewise-constant gust acceleration table (one value per second)."""
    input_port_labels = {}
    output_port_labels = {"dx": 0, "dz": 1}

    def __init__(self, on=True):
        super().__init__()
        self.on = on

    def __len__(self):
        return 0                                    # no input, no algebraic passthrough

    def update(self, t):
        self.outputs.update_from_array(np.array(rc.gust(t, self.on), float))
