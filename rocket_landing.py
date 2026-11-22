#!/usr/bin/env python3
"""Rocket landing in PathSim with custom blocks.

    python3 rocket_landing.py            # runs all cases, prints the check numbers, saves plots to out/
    python3 rocket_landing.py --show     # also opens the plots

Point mass (Lecture 3 model):  calm open loop -> lands on the pad
                               gusts, open loop -> misses the pad (22.3 m)
                               gusts, MPC -> lands on the pad (0.0 m)
Rigid body:                    gusts, MPC + attitude loop
"""
import os, sys, json
import numpy as np
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Function, Integrator
from pathsim.events import ZeroCrossingDown
from pathsim.solvers import RK4

import rocket_core as rc
from rocket_blocks import PointMassRocket, RigidRocket, LandingMPC, AttitudeLoop, GustTable

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)


def build_point_mass(mode="mpc", gusts=True):
    rocket = PointMassRocket()
    mpc = LandingMPC(mode=mode)
    wind = GustTable(on=gusts)
    thrust_norm = Function(lambda fx, fz: np.hypot(fx, fz))
    fuel = Integrator(0.0)
    sc_pos = Scope(labels=["x [m]", "z [m]"])
    sc_thr = Scope(labels=["fx [N]", "fz [N]"])
    sc_fuel = Scope(labels=["fuel [N s]"])

    blocks = [rocket, mpc, wind, thrust_norm, fuel, sc_pos, sc_thr, sc_fuel]
    connections = [
        Connection(rocket["x"], mpc["x"], sc_pos[0]),
        Connection(rocket["z"], mpc["z"], sc_pos[1]),
        Connection(rocket["vx"], mpc["vx"]),
        Connection(rocket["vz"], mpc["vz"]),
        Connection(mpc["fx"], rocket["fx"], thrust_norm[0], sc_thr[0]),
        Connection(mpc["fz"], rocket["fz"], thrust_norm[1], sc_thr[1]),
        Connection(wind["dx"], rocket["dx"]),
        Connection(wind["dz"], rocket["dz"]),
        Connection(thrust_norm[0], fuel[0]),
        Connection(fuel[0], sc_fuel[0]),
    ]
    sim = Simulation(blocks, connections, dt=0.01, Solver=RK4, log=False)
    # stop at ground contact (the planned landing reaches z = 0 at t = 35 s with zero speed)
    sim.add_event(ZeroCrossingDown(func_evt=lambda t: rocket.engine.state[1] + 1e-3, func_act=lambda t: sim.stop()))
    return sim, dict(rocket=rocket, mpc=mpc, fuel=fuel, pos=sc_pos, thr=sc_thr)


def build_rigid(mode="mpc", gusts=True):
    # start pointed along the first planned burn, as a real vehicle would be
    limits = dict(fmax=rc.FMAX_PLAN_RIGID, tilt_max=rc.TILT_MAX, tilt_land=rc.TILT_LAND, slew=rc.SLEW_RIGID)
    _, _, F, _, _ = rc.min_fuel_plan(rc.P0, rc.V0, rc.K, **limits)
    rocket = RigidRocket(theta0=rc.burn_direction(F))
    mpc = LandingMPC(mode=mode, max_delay=rc.MAX_DELAY, **limits)
    att = AttitudeLoop()
    wind = GustTable(on=gusts)
    fuel = Integrator(0.0)
    sc_pos = Scope(labels=["x [m]", "z [m]"])
    sc_att = Scope(labels=["theta [rad]", "theta_ref [rad]", "delta [rad]"])
    sc_act = Scope(labels=["T [N]", "tau_rcs [N m]"])

    blocks = [rocket, mpc, att, wind, fuel, sc_pos, sc_att, sc_act]
    connections = [
        Connection(rocket["x"], mpc["x"], sc_pos[0]),
        Connection(rocket["z"], mpc["z"], sc_pos[1]),
        Connection(rocket["vx"], mpc["vx"]),
        Connection(rocket["vz"], mpc["vz"]),
        Connection(rocket["theta"], att["theta"], sc_att[0]),
        Connection(rocket["omega"], att["omega"]),
        Connection(mpc["fx"], att["fx"]),
        Connection(mpc["fz"], att["fz"]),
        Connection(mpc["theta_ref"], att["theta_ref"], sc_att[1]),
        Connection(att["T"], rocket["T"], fuel[0], sc_act[0]),
        Connection(att["delta"], rocket["delta"], sc_att[2]),
        Connection(att["tau_rcs"], rocket["tau_rcs"], sc_act[1]),
        Connection(wind["dx"], rocket["dx"]),
        Connection(wind["dz"], rocket["dz"]),
    ]
    sim = Simulation(blocks, connections, dt=0.01, Solver=RK4, log=False)
    sim.add_event(ZeroCrossingDown(func_evt=lambda t: rocket.engine.state[1] + 1e-3, func_act=lambda t: sim.stop()))
    return sim, dict(rocket=rocket, mpc=mpc, fuel=fuel, pos=sc_pos, att=sc_att, act=sc_act)


def report(name, sim, b, rigid=False):
    s = b["rocket"].engine.state
    x, z = s[0], s[1]
    vx, vz = (s[3], s[4]) if rigid else (s[2], s[3])
    fuel = float(np.ravel(b["fuel"].engine.state)[0])
    ms = [e["solve_ms"] for e in b["mpc"].planner.log]
    row = dict(case=name, t_end=round(sim.time, 2), x_end=round(float(x), 2), z_end=round(float(z), 3),
               speed_end=round(float(np.hypot(vx, vz)), 2), fuel=round(fuel, 1),
               solves=len(ms), median_solve_ms=round(float(np.median(ms)), 1) if ms else None)
    if rigid:
        row["tilt_end_deg"] = round(float(np.degrees(s[2])), 1)
    print(row)
    return row


if __name__ == "__main__":
    rows, traces = [], {}
    for name, mode, gusts in [("calm, open loop", "open_loop", False), ("gusts, open loop", "open_loop", True),
                              ("gusts, MPC", "mpc", True)]:
        sim, b = build_point_mass(mode, gusts)
        sim.run(duration=rc.K)
        rows.append(report(name, sim, b)); traces[name] = b["pos"].read()
    sim, b = build_rigid("mpc", True)
    sim.run(duration=rc.K + rc.MAX_DELAY + 1)
    rows.append(report("rigid body, gusts, MPC", sim, b, rigid=True)); traces["rigid body, gusts, MPC"] = b["pos"].read()
    json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=1)

    # trajectories
    fig, ax = plt.subplots(figsize=(7, 6))
    zz = np.linspace(0, 110, 50)
    ax.fill_betweenx(zz, zz / np.tan(rc.ALPHA), 130, color="0.92"); ax.fill_betweenx(zz, -zz / np.tan(rc.ALPHA), -60, color="0.92")
    for name, (t, d) in traces.items():
        ax.plot(d[0], d[1], label=name)
    ax.plot(0, 0, "k^", ms=10); ax.set_xlim(-60, 120); ax.set_ylim(-2, 110); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.legend(); ax.set_title("Rocket landing in PathSim (shaded: outside the glide cone)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trajectories.png"), dpi=130)
    if "--show" in sys.argv:
        plt.show()
