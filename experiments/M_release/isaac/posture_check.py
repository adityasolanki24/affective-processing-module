"""Expressive posture check: verify joint signs from rendered stills, and stability while each posture is held.

    $ISAAC_PY experiments/M_release/isaac/posture_check.py --robot humanoid|nerva --out out/posture

For each descriptor at full strength (and the neutral pose) the robot stands 2 s, moves into the posture at the
adapter's rate limit, holds it for 4 s, and front and side stills are rendered. T1 then walks 0.2 m/s for 6 s
with the guarded and with the open posture. Tilt p95, falls and trips are recorded for every condition.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--robot", choices=("humanoid", "nerva"), required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for ext in ("isaacsim.sensors.camera", "isaacsim.storage.native"):
    enable_extension(ext)

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import UsdLux  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from apm_isaac import exit_on_error, hard_exit  # noqa: E402
from apm_isaac.posture import RateLimited, nerva_offsets, t1_offsets  # noqa: E402
from apm_isaac.world import _look_at  # noqa: E402

exit_on_error()
a.out.mkdir(parents=True, exist_ok=True)
DT = 0.02
ZERO = {"gaze_yaw": 0.0, "gaze_pitch": 0.0, "guard": 0.0, "openness": 0.0, "tilt": 0.0, "side_on": 0.0}
POSES = {"neutral": {}, "guard": {"guard": 1.0}, "open": {"openness": 1.0}, "tilt": {"tilt": 1.0},
         "side_on": {"side_on": 1.0}, "gaze_left": {"gaze_yaw": 0.4}, "gaze_up": {"gaze_pitch": 0.25}}

if a.robot == "humanoid":
    world = World(stage_units_in_meters=1.0, physics_dt=0.002, rendering_dt=DT)
    world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0)
    from apm_isaac.booster import BoosterPolicyBody
    body = BoosterPolicyBody(world, "/World/R", (0.0, 0.0), 0.0, 1, ROOT / "experiments/M_release/policies/"
                             "t1_booster_candidate")
    views = {"front": ((2.4, 0.0, 1.1), (0.0, 0.0, 0.85)), "side": ((0.0, -2.4, 1.1), (0.0, 0.0, 0.85))}
    limiter = RateLimited(11, 0.6)
else:
    world = World(stage_units_in_meters=1.0, physics_dt=0.002, rendering_dt=DT)
    world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0)
    from apm_isaac.nerva import NervaIsaac
    body = NervaIsaac(world, "/World/R", (0.0, 0.0), 0.0, 1, Path(os.environ.get("NERVA_ROOT", "/work/NERVA")),
                      Path(os.environ.get("ODP_ROOT", "/work/odp")),
                      Path(os.environ.get("APM_CACHE", "/work/apm_cache")) / "nerva_usd", 0.002, head_channel=True)
    views = {"front": ((0.9, 0.0, 0.40), (0.0, 0.0, 0.28)), "side": ((0.0, -0.9, 0.40), (0.0, 0.0, 0.28))}
    limiter = RateLimited(4, 0.75)
UsdLux.DomeLight.Define(world.stage, "/World/Light").CreateIntensityAttr(1500.0)
cam = Camera("/World/PoseCam", resolution=(720, 720))
world.reset()
body.post_reset()
cam.initialize()
cam.set_focal_length(2.4)


def drive(offsets: np.ndarray, v: float = 0.0) -> dict:
    if a.robot == "humanoid":
        body.upper_offset = offsets
        body.apply(v, 0.0)
    else:
        from apm_express.adapter import MotionRequest
        body.drive(MotionRequest("forward" if v > 0 else "rest", 0.0, tuple(float(x) for x in offsets)))
    return body.advance()


def still(name: str) -> None:
    for view, (eye, aim) in views.items():
        x, y, _ = body.pose()
        e = np.array(eye) + np.array([x, y, 0.0])
        cam.set_world_pose(position=e, orientation=_look_at(e, np.array(aim) + np.array([x, y, 0.0])))
        for _ in range(3):
            world.render()
        frame = np.asarray(cam.get_rgba())
        if frame.size:
            imageio.imwrite(a.out / f"{a.robot}_{name}_{view}.png", frame[..., :3].astype(np.uint8))


results = {}
conditions = [(n, p, 0.0) for n, p in POSES.items()]
if a.robot == "humanoid":
    conditions += [("walk_guard", {"guard": 1.0}, 0.2), ("walk_open", {"openness": 1.0}, 0.2),
                   ("walk_neutral", {}, 0.2)]
for name, pose, v in conditions:
    world.reset()
    body.post_reset()
    limiter.value[:] = 0.0
    g = {**ZERO, **pose}
    target = t1_offsets(g) if a.robot == "humanoid" else nerva_offsets(g)
    tilts, fell, trips0 = [], False, getattr(body, "trips", 0)
    for k in range(int((2.0 + 6.0) / DT)):
        t = k * DT
        off = limiter.step(target if t >= 2.0 else np.zeros_like(target), DT, body.tilt() < math.radians(25))
        row = drive(off, v if t >= 2.0 else 0.0)
        tilts.append(math.degrees(body.tilt()))
        if body.fallen(row):
            fell = True
            break
        if v == 0.0 and abs(t - 7.0) < DT / 2:
            still(name)
    results[name] = {"target": np.round(target, 3).tolist(), "tilt_p95_deg": float(np.percentile(tilts, 95)),
                     "tilt_max_deg": float(np.max(tilts)), "fell": fell,
                     "trips": int(getattr(body, "trips", 0) - trips0)}
    print("POSE", name, json.dumps(results[name]), flush=True)
(a.out / f"posture_{a.robot}.json").write_text(json.dumps(results, indent=1))
print("POSTURE DONE", a.robot, flush=True)
hard_exit(0)
