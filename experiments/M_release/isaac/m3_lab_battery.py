"""M3 gate: the operating envelope of an Isaac Lab-trained controller, deployed the way APM deploys it
(`LabPolicyBody`).

    $ISAAC_PY experiments/M_release/isaac/m3_lab_battery.py --body t1 --policy out/t1_train --seeds 10
        --out out/t1_battery

The physics matches training: 200 Hz, 50 Hz control, the trained PD gains and armature, self-collision off, 8/4
solver iterations.

**Conditions** (12 s each, the body's tilt supervisor on):
- speeds 0–0.5 m/s straight;
- turns at ±0.5 and ±1 rad/s (at 0.2 m/s);
- stop/start every 1.5 s;
- repeated pushes every 1–2 s at 0.1, 0.3, 0.5 and 0.8 m/s while walking, turning and standing;
- observation noise.

Reported per condition: falls, trips, speed and turn-rate tracking error, and the 95th-percentile tilt. The
structure is the same as programme L's A7, so the envelopes can be compared.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--body", choices=("t1", "nerva"), required=True)
ap.add_argument("--policy", type=Path, required=True)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
app = AppLauncher({"headless": True}).app

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac.robots import LabPolicyBody  # noqa: E402

BODIES = {
    "t1": {"usd": get_assets_root_path() + "/Isaac/Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd", "z": 0.70,
           "gains": {".*_Hip_.*": (100.0, 3.0), ".*_Knee_Pitch": (150.0, 4.0), "Waist": (100.0, 3.0),
                     ".*_Ankle_.*": (40.0, 2.0), ".*_Shoulder_.*": (40.0, 2.0), ".*_Elbow_.*": (40.0, 2.0),
                     "AAHead_yaw": (20.0, 1.0), "Head_pitch": (20.0, 1.0)}, "v_max": 0.5},
}
B = BODIES[a.body]


def conditions() -> dict:
    c: dict = {f"speed_{v}": {"v": v} for v in (0.0, 0.1, 0.2, 0.3, 0.5) if v <= B["v_max"]}
    c.update({f"turn_{w}": {"v": 0.2, "w": w} for w in (-1.0, -0.5, 0.5, 1.0)})
    c["stop_start"] = {"v": 0.3, "stop_start": 1.5}
    for m in (0.1, 0.3, 0.5, 0.8):
        c[f"push_walk_{m}"] = {"v": 0.2, "push": m}
        c[f"push_turn_{m}"] = {"v": 0.2, "w": 0.8, "push": m}
        c[f"push_stand_{m}"] = {"v": 0.0, "push": m}
    return c


world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=0.02)
world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0)
body = LabPolicyBody(world, "/World/Robot", B["usd"], (0.0, 0.0), 0.0, 0, a.policy, B["z"], physics_dt=0.005)
root = str(body.art.prim_path)
art = PhysxSchema.PhysxArticulationAPI.Apply(world.stage.GetPrimAtPath(root))
art.CreateEnabledSelfCollisionsAttr(False)
art.CreateSolverPositionIterationCountAttr(8)
art.CreateSolverVelocityIterationCountAttr(4)
for p in world.stage.Traverse():
    if str(p.GetPath()).startswith("/World/Robot") and p.IsA(UsdPhysics.RevoluteJoint):
        PhysxSchema.PhysxJointAPI.Apply(p).CreateArmatureAttr(0.01)
world.reset()
rows = []
for name, cfg in conditions().items():
    for seed in range(a.seeds):
        rng = np.random.default_rng([seed, 2026, 1019])
        world.reset()
        body._init_common(seed)
        body.post_reset(B["gains"])
        nxt = rng.uniform(1.0, 2.0)
        fell, tilts, vf, wz = False, [], [], []
        for k in range(int(12.0 / 0.02)):
            t = k * 0.02
            v, w = cfg.get("v", 0.0), cfg.get("w", 0.0)
            if cfg.get("stop_start") and int(t / cfg["stop_start"]) % 2 == 1:
                v, w = 0.0, 0.0
            body.apply(v, w)
            if cfg.get("push") and t >= nxt:
                ang = rng.uniform(0, 2 * math.pi)
                body.push(cfg["push"] * math.cos(ang), cfg["push"] * math.sin(ang))
                nxt = t + rng.uniform(1.0, 2.0)
            r = body.advance()
            tilts.append(r["tilt"])
            if body.fallen(r):
                fell = True
                break
            if t >= 6.0:
                _, _, yaw = body.pose()
                vel = np.asarray(body.art.get_linear_velocity(), float)
                vf.append(vel[0] * math.cos(yaw) + vel[1] * math.sin(yaw))
                wz.append(float(np.asarray(body.art.get_angular_velocity())[2]))
        track = not cfg.get("stop_start") and not cfg.get("push") and not fell
        rows.append({"cond": name, "seed": seed, "fell": fell, "trips": body.trips, "max_tilt": max(tilts),
                     "v_err": float(np.mean(vf)) - cfg.get("v", 0.0) if track else None,
                     "w_err": float(np.mean(wz)) - cfg.get("w", 0.0) if track else None})
    rr = [r for r in rows if r["cond"] == name]
    ve = [r["v_err"] for r in rr if r["v_err"] is not None]
    we = [r["w_err"] for r in rr if r["w_err"] is not None]
    print(f"{name:18} falls {sum(r['fell'] for r in rr)}/{len(rr)} trips {np.mean([r['trips'] for r in rr]):.2f} "
          f"tilt95 {np.percentile([r['max_tilt'] for r in rr], 95):.1f} v_err {np.mean(ve) if ve else None} "
          f"w_err {np.mean(we) if we else None}", flush=True)
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "battery.json").write_text(json.dumps({"body": a.body, "policy": str(a.policy), "rows": rows}, indent=1))
from apm_isaac import hard_exit  # noqa: E402

hard_exit(0)
