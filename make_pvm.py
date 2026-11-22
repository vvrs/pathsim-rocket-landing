#!/usr/bin/env python3
"""Write the PathView model files (.pvm) for the rocket landing.

    python3 make_pvm.py    -> rocket_point_mass.pvm, rocket_rigid_body.pvm

PathView draws built-in block types, so each custom block from rocket_blocks.py
appears as the built-in block it subclasses (ODE, Wrapper, Function, Source),
named after it, with its functions in the model's code panel. That code panel is
rocket_core.py pasted in verbatim, plus a few lines that set the model up.
"""
import json, os, uuid
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = open(os.path.join(HERE, "rocket_core.py"), encoding="utf-8").read()
GREY = "#969696"


def _id(): return str(uuid.uuid4())


def node(kind, name, x, y, params, ins=(), outs=()):
    nid = _id()
    port = lambda d, i, n: {"id": f"{nid}-{d}-{i}", "nodeId": nid, "name": n, "direction": d, "index": i, "color": GREY}
    return {"id": nid, "type": kind, "name": name, "position": {"x": x, "y": y},
            "inputs": [port("input", i, n) for i, n in enumerate(ins)],
            "outputs": [port("output", i, n) for i, n in enumerate(outs)],
            "params": {**params, "_rotation": 0}}


def wire(a, ai, b, bi):
    return {"id": _id(), "sourceNodeId": a["id"], "sourcePortIndex": ai, "targetNodeId": b["id"], "targetPortIndex": bi}


def pvm(name, nodes, conns, setup, duration, events=()):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return {"version": "1.0.0", "metadata": {"created": now, "modified": now, "name": name},
            "graph": {"nodes": nodes, "connections": conns}, "events": list(events),
            "codeContext": {"code": CORE + "\n\n" + setup},
            "simulationSettings": {"duration": str(duration), "dt": "0.01", "solver": "RK4", "adaptive": False,
                                   "atol": "1e-6", "rtol": "1e-4", "ftol": "1e-10", "dt_min": None, "dt_max": None,
                                   "ghostTraces": 0, "plotResults": True}}


def touchdown_event():
    return {"id": _id(), "type": "pathsim.events.ZeroCrossingDown", "name": "Landed",
            "position": {"x": 1180, "y": 500}, "params": {"func_evt": "touchdown", "func_act": "stop_sim"}}


# ---------------------------------------------------------------------------------------------
def point_mass():
    setup = '''# ==== MODEL SETUP (edit here) ==============================================================
MODE = "mpc"      # "mpc": re-solve every second   |   "open_loop": fly the t = 0 plan blind
GUSTS = True      # wind gust sequence on/off

planner = ConvexLandingPlanner(mode=MODE)               # the "Convex MPC" block calls this every 1 s
s0 = np.array([P0[0], P0[1], V0[0], V0[1]])             # x, z, vx, vz

def gust_x(t): return gust(t, GUSTS)[0]
def gust_z(t): return gust(t, GUSTS)[1]

def touchdown(t): return rocket.engine.state[1] + 1e-3  # z of the "Rocket" block (+1 mm)
def stop_sim(t): sim.stop()
'''
    mpc = node("Wrapper", "MPC", 300, 260, {"func": "planner", "T": "1.0"},
               ins=["x", "z", "vx", "vz"], outs=["fx", "fz", "theta_ref"])
    rocket = node("ODE", "Rocket", 640, 260, {"func": "point_mass_rhs", "initial_value": "s0"},
                  ins=["fx", "fz", "dx", "dz"], outs=["x", "z", "vx", "vz"])
    gx = node("Source", "Gust x", 420, 400, {"func": "gust_x"}, outs=["out"])
    gz = node("Source", "Gust z", 420, 480, {"func": "gust_z"}, outs=["out"])
    norm = node("Function", "|thrust|", 640, 80, {"func": "lambda fx, fz: np.hypot(fx, fz)"}, ins=["fx", "fz"], outs=["|f|"])
    fuel = node("Integrator", "Fuel", 880, 80, {"initial_value": "0.0"}, ins=["in"], outs=["out"])
    sc_pos = node("Scope", "Position", 960, 230, {"labels": '["x [m]", "z [m]"]'}, ins=["x", "z"])
    sc_vel = node("Scope", "Velocity", 960, 360, {"labels": '["vx [m/s]", "vz [m/s]"]'}, ins=["vx", "vz"])
    sc_thr = node("Scope", "Thrust", 640, -60, {"labels": '["fx [N]", "fz [N]"]'}, ins=["fx", "fz"])
    sc_fuel = node("Scope", "Fuel used", 1100, 80, {"labels": '["fuel [N s]"]'}, ins=["fuel"])
    nodes = [mpc, rocket, gx, gz, norm, fuel, sc_pos, sc_vel, sc_thr, sc_fuel]
    conns = [wire(mpc, 0, rocket, 0), wire(mpc, 1, rocket, 1), wire(gx, 0, rocket, 2), wire(gz, 0, rocket, 3),
             *[wire(rocket, i, mpc, i) for i in range(4)],
             wire(rocket, 0, sc_pos, 0), wire(rocket, 1, sc_pos, 1), wire(rocket, 2, sc_vel, 0), wire(rocket, 3, sc_vel, 1),
             wire(mpc, 0, norm, 0), wire(mpc, 1, norm, 1), wire(mpc, 0, sc_thr, 0), wire(mpc, 1, sc_thr, 1),
             wire(norm, 0, fuel, 0), wire(fuel, 0, sc_fuel, 0)]
    return pvm("Rocket landing: point mass + convex MPC", nodes, conns, setup, 35, [touchdown_event()])


def rigid_body():
    setup = '''# ==== MODEL SETUP (edit here) ==============================================================
GUSTS = True
LIMITS = dict(fmax=FMAX_PLAN_RIGID, tilt_max=TILT_MAX, tilt_land=TILT_LAND, slew=SLEW_RIGID)

planner = ConvexLandingPlanner(mode="mpc", max_delay=MAX_DELAY, **LIMITS)
_, _, _F0, _, _ = min_fuel_plan(P0, V0, K, **LIMITS)
s0 = np.array([P0[0], P0[1], burn_direction(_F0), V0[0], V0[1], 0.0])   # start pointed along the first burn

def attitude_loop(fx, fz, th_ref, th, om):
    return np.array(attitude_control(fx, fz, th_ref, th, om))

def gust_x(t): return gust(t, GUSTS)[0]
def gust_z(t): return gust(t, GUSTS)[1]

def touchdown(t): return rocket.engine.state[1] + 1e-3
def stop_sim(t): sim.stop()
'''
    mpc = node("Wrapper", "MPC", 160, 260, {"func": "planner", "T": "1.0"},
               ins=["x", "z", "vx", "vz"], outs=["fx", "fz", "theta_ref"])
    att = node("Function", "Attitude", 440, 260, {"func": "attitude_loop"},
               ins=["fx", "fz", "theta_ref", "theta", "omega"], outs=["T", "delta", "tau_rcs"])
    rocket = node("ODE", "Rocket", 740, 260, {"func": "rigid_body_rhs", "initial_value": "s0"},
                  ins=["T", "delta", "tau_rcs", "dx", "dz"], outs=["x", "z", "theta", "vx", "vz", "omega"])
    gx = node("Source", "Gust x", 520, 420, {"func": "gust_x"}, outs=["out"])
    gz = node("Source", "Gust z", 520, 500, {"func": "gust_z"}, outs=["out"])
    fuel = node("Integrator", "Fuel", 740, 60, {"initial_value": "0.0"}, ins=["T"], outs=["out"])
    sc_pos = node("Scope", "Position", 1060, 200, {"labels": '["x [m]", "z [m]"]'}, ins=["x", "z"])
    sc_att = node("Scope", "Tilt", 1060, 340, {"labels": '["theta [rad]", "theta_ref [rad]"]'}, ins=["theta", "theta_ref"])
    sc_act = node("Scope", "Actuators", 440, 60, {"labels": '["T [N]", "delta [rad]", "tau_rcs [N m]"]'}, ins=["T", "delta", "tau_rcs"])
    sc_fuel = node("Scope", "Fuel used", 980, 60, {"labels": '["fuel [N s]"]'}, ins=["fuel"])
    nodes = [mpc, att, rocket, gx, gz, fuel, sc_pos, sc_att, sc_act, sc_fuel]
    conns = [wire(mpc, 0, att, 0), wire(mpc, 1, att, 1), wire(mpc, 2, att, 2),
             wire(rocket, 2, att, 3), wire(rocket, 5, att, 4),
             wire(att, 0, rocket, 0), wire(att, 1, rocket, 1), wire(att, 2, rocket, 2),
             wire(gx, 0, rocket, 3), wire(gz, 0, rocket, 4),
             wire(rocket, 0, mpc, 0), wire(rocket, 1, mpc, 1), wire(rocket, 3, mpc, 2), wire(rocket, 4, mpc, 3),
             wire(rocket, 0, sc_pos, 0), wire(rocket, 1, sc_pos, 1), wire(rocket, 2, sc_att, 0), wire(mpc, 2, sc_att, 1),
             wire(att, 0, sc_act, 0), wire(att, 1, sc_act, 1), wire(att, 2, sc_act, 2),
             wire(att, 0, fuel, 0), wire(fuel, 0, sc_fuel, 0)]
    return pvm("Rocket landing: rigid body + attitude loop + convex MPC", nodes, conns, setup, 44, [touchdown_event()])


if __name__ == "__main__":
    for fname, model in [("rocket_point_mass.pvm", point_mass()), ("rocket_rigid_body.pvm", rigid_body())]:
        with open(os.path.join(HERE, fname), "w", encoding="utf-8") as f:
            json.dump(model, f, indent=1)
        print("wrote", fname)
