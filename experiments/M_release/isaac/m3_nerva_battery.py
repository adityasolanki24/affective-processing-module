"""M3 gate and cross-physics check: NERVA's frozen neutral controller on one battery, in Isaac Sim and in MuJoCo.

    $ISAAC_PY experiments/M_release/isaac/m3_nerva_battery.py --engine isaac --seeds 10 --out out/nerva_battery
    $ISAAC_PY experiments/M_release/isaac/m3_nerva_battery.py --engine mujoco --seeds 10 --out out/nerva_battery

Both engines use NERVA's deployed stack:
- the seven trained commands (H's `MotionRequest`) and head offsets;
- NERVA's training observation noise;
- NERVA's `SafetyMonitor` with H's admitted limits (neutral fallback; a trip is counted once per episode of
  fallback).

Isaac: `apm_isaac.nerva.NervaIsaac` (the M3-fixed import, no backlash). MuJoCo: `apm_express.biped.NervaBiped` (the
validated backlash model).

**Conditions** (12 s each, after 1 s at rest):
- rest;
- forward, backward;
- turn left, turn right;
- stop/start (forward and rest alternating every 1.5 s);
- head motion at rest (head yaw ±0.4 rad at 0.25 Hz);
- repeated pushes every 1–2 s in random directions at 0.05, 0.10, 0.15 and 0.20 m/s, while walking and at rest.

Reported per condition: falls (tilt > 45°), monitor trips, mean forward speed and yaw rate over the last 6 s
(heading frame), 95th-percentile tilt. This is a simulation envelope, not a certification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=("isaac", "mujoco"), required=True)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("NERVA_ROOT", "/work/NERVA")
sys.path.insert(0, os.environ["NERVA_ROOT"])
sys.path.insert(0, os.environ.get("ODP_ROOT", "/work/odp"))  # Open Duck Playground (NERVA's MuJoCo backend)
T_RUN, T_SETTLE, DT = 12.0, 1.0, 0.02


def conditions() -> dict[str, dict]:
    c: dict[str, dict] = {"rest": {"cmd": "rest"}, "forward": {"cmd": "forward"}, "backward": {"cmd": "backward"},
                          "turn_left": {"cmd": "turn_left"}, "turn_right": {"cmd": "turn_right"},
                          "stop_start": {"cmd": "forward", "stop_start": 1.5}, "head_motion": {"cmd": "rest",
                                                                                                "head": True}}
    for m in (0.05, 0.10, 0.15, 0.20):
        c[f"push_walk_{m}"] = {"cmd": "forward", "push": m}
        c[f"push_rest_{m}"] = {"cmd": "rest", "push": m}
    return c


if a.engine == "isaac":
    from isaaclab.app import AppLauncher
    app = AppLauncher({"headless": True}).app

import numpy as np  # noqa: E402

from apm_express.adapter import MotionRequest  # noqa: E402
from apm_express.safety import MonitorLimits, SafetyMonitor  # noqa: E402

CH = json.loads((REPO / "experiments" / "J_validation_transfer" / "configs" / "H_admitted_interface.json")
                .read_text())["interface"]


def limits() -> MonitorLimits:
    return MonitorLimits(joint_margin_rad=CH.get("joint_margin_rad"), saturation_steps=CH.get("saturation_steps"),
                         **{k: CH[k] for k in ("stop_tilt_deg", "stop_angular_speed") if k in CH})


if a.engine == "isaac":
    from isaacsim.core.api import World

    from apm_isaac.nerva import NervaIsaac
    world = World(stage_units_in_meters=1.0, physics_dt=0.002, rendering_dt=0.02)
    world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0)
    body = NervaIsaac(world, "/World/Robot", (0.0, 0.0), 0.0, 0, Path(os.environ["NERVA_ROOT"]),
                      Path(os.environ.get("ODP_ROOT", "/work/odp")), Path("/work/apm_cache/nerva_usd"),
                      head_channel=True, limits=limits())
    world.reset()
    body.post_reset()
    print("measured", json.dumps(body.measured, default=str)[:1500], flush=True)

    class Episode:
        def __init__(self, seed: int):
            world.reset()
            body._seed = seed
            body._init_common(seed)
            body.monitor = SafetyMonitor(limits())
            body.post_reset()
            self.t = 0.0

        def step(self, req) -> dict:
            body.drive(req)
            r = body.advance()
            x, y, yaw = body.pose()
            v = np.asarray(body.art.get_linear_velocity(), float)
            w = float(np.asarray(body.art.get_angular_velocity(), float)[2])
            return {"tilt": r["tilt"], "fell": body.fallen(r), "vf": v[0] * math.cos(yaw) + v[1] * math.sin(yaw),
                    "wz": w, "trips": body.trips}

        def push(self, dvx: float, dvy: float) -> None:
            body.push(dvx, dvy)
else:
    from apm_express.adapter import NEUTRAL_REQUEST
    from apm_express.biped import NervaBiped

    class Episode:  # type: ignore[no-redef]
        def __init__(self, seed: int):
            self.c = NervaBiped(policy=CH["policy"], seed=seed, head_channel=True, max_head_rate=1.5,
                                max_pitch_rate=2.0)
            self.mon = SafetyMonitor(limits())
            self.trips, self.was = 0, False
            self.prev = None

        def step(self, req) -> dict:
            tripped = bool(self.mon.update(self.c.t, **self.c.physical()))
            if tripped and not self.was:
                self.trips += 1
            self.was = tripped
            self.c.apply(NEUTRAL_REQUEST if tripped else req)
            r = self.c.advance()
            return {"tilt": r["tilt"], "fell": bool(r["tilt"] > 45.0 or not r["finite"]),
                    "vf": r["vx_world"] * math.cos(r["yaw"]) + r["vy_world"] * math.sin(r["yaw"]),
                    "wz": float(self.c.sim.data.qvel[5]), "trips": self.trips}

        def push(self, dvx: float, dvy: float) -> None:
            self.c.push(dvx, dvy)

TRAINED = {"forward": (0.074, 0.0), "backward": (-0.074, 0.0), "turn_left": (0.0, 0.6), "turn_right": (0.0, -0.6),
           "rest": (0.0, 0.0)}
rows = []
for name, cfg in conditions().items():
    for seed in range(a.seeds):
        rng = np.random.default_rng([seed, 2026, 1019])
        ep = Episode(seed)
        nxt = T_SETTLE + rng.uniform(1.0, 2.0)
        tilts, vf, wz, fell, last = [], [], [], False, {}
        for k in range(int((T_SETTLE + T_RUN) / DT)):
            t = k * DT
            cmd = "rest" if t < T_SETTLE else cfg["cmd"]
            if cfg.get("stop_start") and t >= T_SETTLE and int((t - T_SETTLE) / cfg["stop_start"]) % 2 == 1:
                cmd = "rest"
            head = (0.0, 0.0, 0.4 * math.sin(2 * math.pi * 0.25 * t), 0.0) if cfg.get("head") and t >= T_SETTLE \
                else (0.0, 0.0, 0.0, 0.0)
            if cfg.get("push") and t >= nxt:
                ang = rng.uniform(0, 2 * math.pi)
                ep.push(cfg["push"] * math.cos(ang), cfg["push"] * math.sin(ang))
                nxt = t + rng.uniform(1.0, 2.0)
            last = ep.step(MotionRequest(cmd, 0.0, head))
            tilts.append(last["tilt"])
            if last["fell"]:
                fell = True
                break
            if t >= T_SETTLE + T_RUN - 6.0:
                vf.append(last["vf"])
                wz.append(last["wz"])
        track = not cfg.get("stop_start") and not cfg.get("push") and not fell
        v0, w0 = TRAINED[cfg["cmd"]]
        rows.append({"cond": name, "seed": seed, "fell": fell, "trips": last.get("trips", 0),
                     "tilt95": float(np.percentile(tilts, 95)), "max_tilt": float(max(tilts)),
                     "v": float(np.mean(vf)) if track else None, "w": float(np.mean(wz)) if track else None,
                     "v_cmd": v0, "w_cmd": w0})
    rr = [r for r in rows if r["cond"] == name]
    vs = [r["v"] for r in rr if r["v"] is not None]
    ws = [r["w"] for r in rr if r["w"] is not None]
    print(f"{name:16} falls {sum(r['fell'] for r in rr)}/{len(rr)} trips {np.mean([r['trips'] for r in rr]):.2f} "
          f"tilt95 {np.mean([r['tilt95'] for r in rr]):.1f} v {np.mean(vs) if vs else float('nan'):.4f} "
          f"w {np.mean(ws) if ws else float('nan'):.3f}", flush=True)
a.out.mkdir(parents=True, exist_ok=True)
(a.out / f"nerva_battery_{a.engine}.json").write_text(json.dumps({"engine": a.engine, "rows": rows}, indent=1))
if a.engine == "isaac":
    from apm_isaac import hard_exit
    hard_exit(0)
