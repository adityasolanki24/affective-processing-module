"""M3: validate the wheeled embodiment in Isaac Sim (NVIDIA Nova Carter by default; `--robot reference` is the custom
robot matched to MuJoCo), and the reference robot against MuJoCo on the same command script.

    $ISAAC_PY experiments/M_release/isaac/m3_wheeled.py --out out/m3_wheeled [--robot reference]  (Isaac side)
    python experiments/M_release/isaac/m3_wheeled.py --mujoco --out <dir>        (MuJoCo reference)

**Script** (one run, 50 Hz commands; the (v, ω) tracking controller and supervisor are the body's own):
- rest 2 s;
- straight at 0.3 m/s for 6 s, then stop;
- turn on the spot at 1 rad/s for 4 s;
- arc at (0.2 m/s, 0.5 rad/s) for 6 s;
- stop.

**Recorded** (x, y, yaw at 50 Hz), and from them:
- speed and turn-rate tracking error;
- stopping distance after the stop command;
- IMU at rest: specific force ≈ (0, 0, 9.81).

**Isaac only:**
- a 0.6 m/s shove (the supervisor must trip; there must be no fall);
- the depth eye's range to a 0.5 × 0.5 × 1.2 m box whose face is 1.0 m ahead of the eye (median over its centre
  pixels).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mujoco", action="store_true")
ap.add_argument("--robot", choices=("nova_carter", "reference"), default="nova_carter",
                help="nova_carter: the release's wheeled embodiment; reference: the custom robot matched to MuJoCo")
a, _ = ap.parse_known_args()
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
SCRIPT = [(0.0, 0.0, 2.0), (0.3, 0.0, 6.0), (0.0, 0.0, 3.0), (0.0, 1.0, 4.0), (0.0, 0.0, 2.0), (0.2, 0.5, 6.0),
          (0.0, 0.0, 3.0)]


def drive(robot, rows: list) -> None:
    for v, w, dur in SCRIPT:
        for _ in range(int(round(dur / 0.02))):
            robot.apply(v, w)
            r = robot.advance()
            x, y, yaw = robot.pose()
            rows.append({"t": round(robot.t, 3), "x": x, "y": y, "yaw": yaw, "v_cmd": v, "w_cmd": w,
                         "acc": [float(c) for c in r["acc"]], "tilt": r["tilt"]})


def metrics(rows: list) -> dict:
    import numpy as np
    xs, ys, yaws = (np.array([r[k] for r in rows]) for k in ("x", "y", "yaw"))
    v = np.hypot(np.diff(xs), np.diff(ys)) / 0.02
    w = np.diff(np.unwrap(yaws)) / 0.02
    vc = np.array([r["v_cmd"] for r in rows[1:]])
    wc = np.array([r["w_cmd"] for r in rows[1:]])
    steady = np.array([i > 50 and rows[i]["v_cmd"] == rows[i - 50]["v_cmd"]
                       and rows[i]["w_cmd"] == rows[i - 50]["w_cmd"] for i in range(1, len(rows))])
    out = {"v_err_mean": float(np.mean(np.abs(v - np.abs(vc))[steady & (vc > 0)])),
           "w_err_mean": float(np.mean(np.abs(w - wc)[steady & (wc != 0)])),
           "rest_acc": np.mean([r["acc"] for r in rows[20:90]], axis=0).tolist(),
           "max_tilt": float(max(r["tilt"] for r in rows))}
    i_stop = next(i for i, r in enumerate(rows) if r["t"] > 8.0 + 1e-9)
    i_end = next(i for i, r in enumerate(rows) if r["t"] > 11.0 - 1e-9)
    out["stop_distance"] = float(math.hypot(xs[i_end] - xs[i_stop], ys[i_end] - ys[i_stop]))
    out["final_pose"] = [float(xs[-1]), float(ys[-1]), float(yaws[-1])]
    return out


if a.mujoco:
    from apm_transfer.wheeled_world import WheeledRobot
    bot = WheeledRobot(0)
    rows: list = []
    drive(bot, rows)
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"engine": "mujoco 3.3", "metrics": metrics(rows)}
    (a.out / "wheeled_mujoco.json").write_text(json.dumps({**res, "rows": rows}))
    print(json.dumps(res, indent=1))
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for _e in ("isaacsim.sensors.camera",):
    enable_extension(_e)
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid  # noqa: E402

from apm_isaac.robots import NovaCarterIsaac, WheeledIsaac  # noqa: E402

a.out.mkdir(parents=True, exist_ok=True)
world = World(stage_units_in_meters=1.0, physics_dt=0.002, rendering_dt=0.02)
world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0)
bot = (NovaCarterIsaac if a.robot == "nova_carter" else WheeledIsaac)(world, "/World/Robot", (0.0, 0.0), 0.0, seed=0)
world.reset()
if hasattr(bot, "post_reset"):
    bot.post_reset()
print("measured", json.dumps(getattr(bot, "measured", {})), flush=True)
rows = []
drive(bot, rows)
res: dict = {"engine": "isaac sim 5.1 physx", "robot": a.robot, "metrics": metrics(rows),
             "dof_names": list(bot.art.dof_names), "measured": getattr(bot, "measured", None)}
# shove: the supervisor must trip (speed > 0.35 m/s) and the robot must not fall
trips0 = bot.trips
bot.push(0.6, 0.0)
fell = False
for _ in range(100):
    bot.apply(0.0, 0.0)
    r = bot.advance()
    fell |= bot.fallen(r)
res["shove"] = {"tripped": bot.trips > trips0, "fell": fell}
# depth eye range check: a box 1.0 m ahead of the eye
x, y, yaw = bot.pose()
ex, ey = x + bot.eye_offset * math.cos(yaw), y + bot.eye_offset * math.sin(yaw)
world.scene.add(FixedCuboid(prim_path="/World/target_box", name="target_box",
                            position=np.array([ex + 1.25 * math.cos(yaw), ey + 1.25 * math.sin(yaw), 0.6]),
                            scale=np.array([0.5, 0.5, 1.2])))
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402

from apm_transfer.vision import VisionCamera  # noqa: E402

c = VisionCamera(pitch_deg=0.0)
cam = Camera(prim_path=f"{bot.eye_prim}/depth_eye", resolution=(c.width, c.height),
             translation=np.zeros(3), orientation=euler_angles_to_quat(np.zeros(3), degrees=True))
cam.initialize()
cam.set_focal_length(1.0)
cam.set_horizontal_aperture(2 * math.tan(math.radians(c.hfov_deg) / 2))
cam.add_distance_to_image_plane_to_frame()
for _ in range(20):
    world.step(render=True)
d = cam.get_depth()
if d is not None and np.size(d):
    centre = np.asarray(d)[c.height // 2 - 3:c.height // 2 + 3, c.width // 2 - 3:c.width // 2 + 3]
    res["depth_check"] = {"expected_face_m": 1.0, "median_centre_m": float(np.nanmedian(centre)),
                          "shape": list(np.shape(d))}
    np.save(a.out / "depth_frame.npy", np.asarray(d, np.float32))
# a look at the robot itself (release QA)
from PIL import Image  # noqa: E402

from apm_isaac.world import _look_at  # noqa: E402

vc = Camera(prim_path="/World/qa_cam", resolution=(960, 540))
vc.initialize()
vc.set_focal_length(1.8)
for k, (dx, dy) in enumerate(((2.2, 1.2), (-1.5, 2.0))):
    eye = np.array([x + dx, y + dy, 1.2])
    vc.set_world_pose(position=eye, orientation=_look_at(eye, np.array([x, y, 0.3])))
    for _ in range(10):
        world.step(render=True)
    f = vc.get_rgba()
    if f is not None and np.size(f):
        Image.fromarray(np.asarray(f[..., :3], np.uint8)).save(a.out / f"robot_view{k}.png")
(a.out / f"wheeled_isaac_{a.robot}.json").write_text(json.dumps({**res, "rows": rows}))
print(json.dumps(res, indent=1), flush=True)
from apm_isaac import hard_exit  # noqa: E402

hard_exit(0)
