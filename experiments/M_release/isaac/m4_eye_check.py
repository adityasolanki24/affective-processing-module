"""M4 check: what does each robot's depth eye see? (The camera is APM's only view of a person's reach.)

    $ISAAC_PY experiments/M_release/isaac/m4_eye_check.py --body nova_carter --out out/eye
        [--policy-dir DIR]   (humanoid)

A person stands with the capsule surface 0.8 m ahead of the robot's eye on a flat ground plane. Reported:
- the eye's world position and heading, and its height;
- the depth image's median over the centre (expected: about 0.8 m to the person's front);
- the person's pixel coverage;
- `MotionPerception` on a scripted lunge of 0.25 m toward the robot (expected: a reach is detected).

Saved: the depth image (npy, png) and an RGB view from the same pose at a higher resolution.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--body", required=True, choices=("nova_carter", "humanoid", "nerva", "wheeled_reference"))
ap.add_argument("--policy-dir", default=None)
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac import ANIM_KIT_ARGS  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True, "kit_args": ANIM_KIT_ARGS}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
import numpy as np  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from apm_isaac.agents import GAP0, RADIUS, Lunge, find_character  # noqa: E402
from apm_isaac.world import IsaacWorld, SceneSpec  # noqa: E402
from apm_transfer.vision import MotionPerception  # noqa: E402

a.out.mkdir(parents=True, exist_ok=True)
root = get_assets_root_path()
channels = json.loads((REPO / "experiments" / "J_validation_transfer" / "configs" / "H_admitted_interface.json")
                      .read_text())["interface"]
# a plain floor: the stock grid environment
scene = SceneSpec("grid", "/Isaac/Environments/Grid/default_environment.usd", home=(0.0, 0.0, 0.0),
                  stations={"P1": (3.0, 0.0)})
w = IsaacWorld(scene, a.body, seed=1, characters={"P1": find_character(root, "male_adult_construction_05_new")},
               physics_dt=0.005 if a.body == "humanoid" else 0.002, policy_dir=a.policy_dir, nerva_channels=channels)
for _ in range(50):
    w.step(w._driver(0).idle, w._driver(0), False)
# signed drive test: does a forward command move the robot along its own heading?
x0, y0, yaw0 = w.robot.pose()
if a.body != "nerva":
    d0 = w._driver(0)
    for _ in range(100):
        w.step((0.15, 0.0), d0, False)
    for _ in range(50):
        w.step(d0.idle, d0, False)
x1, y1, _ = w.robot.pose()
print("drive_test", {"heading_deg": round(math.degrees(yaw0), 1), "dx": round(x1 - x0, 3), "dy": round(y1 - y0, 3),
                     "along_heading": round((x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0), 3)}, flush=True)
x, y, yaw = w.robot.pose()
eye_x = x + w.robot.eye_offset * math.cos(yaw)
person = w.people["P1"]
person.xy = np.array([eye_x + 0.8 + RADIUS, y])
drv = w._driver(0)
for _ in range(25):
    w.step(drv.idle, drv, False)
eye_w = UsdGeom.XformCache().GetLocalToWorldTransform(w.world.stage.GetPrimAtPath(w.robot.eye_prim))
# where is the person's visual character?
from pxr import Usd  # noqa: E402

vis = w.world.stage.GetPrimAtPath(person.vis_path)  # type: ignore[attr-defined]
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(vis).ComputeAlignedRange()
print("character", person.vis_path, "animated", getattr(person, "animated", None),  # type: ignore[attr-defined]
      "visibility", UsdGeom.Imageable(vis).ComputeVisibility(), "bbox", list(bb.GetMin()), list(bb.GetMax()),
      "xform", list(UsdGeom.XformCache().GetLocalToWorldTransform(vis).ExtractTranslation()),
      "person_xy", person.xy.tolist(), flush=True)
tp = Camera(prim_path="/World/third_person", resolution=(800, 450))
tp.initialize()
tp.set_focal_length(1.8)
from apm_isaac.world import _look_at  # noqa: E402

tp_eye = np.array([x + 0.6, y - 3.0, 1.5])
tp.set_world_pose(position=tp_eye, orientation=_look_at(tp_eye, np.array([x + 0.6, y, 0.5])))
for _ in range(5):
    w.world.step(render=True)
f3 = tp.get_rgba()
if f3 is not None and np.size(f3):
    Image.fromarray(np.asarray(f3[..., :3], np.uint8)).save(a.out / f"{a.body}_third_person.png")
rep: dict = {"body": a.body, "eye_prim": w.robot.eye_prim, "eye_offset": w.robot.eye_offset,
             "robot_pose": [x, y, yaw], "eye_usd_translation": list(eye_w.ExtractTranslation())}
cam_pos, cam_q = w._eye_cam.get_world_pose(camera_axes="world")
rep["eye_camera_world_position"] = np.round(np.asarray(cam_pos, float), 3).tolist()
from apm_isaac.robots import quat_to_rot  # noqa: E402

rep["eye_camera_view_dir_world"] = np.round(quat_to_rot(np.asarray(cam_q, float))[:, 0], 3).tolist()
rep["robot_heading_dir_world"] = [round(math.cos(yaw), 3), round(math.sin(yaw), 3)]
d = w.depth()
c = w.vision_cfg
ctr = d[c.height // 2 - 4:c.height // 2 + 4, c.width // 2 - 4:c.width // 2 + 4]
rep["depth_centre_median"] = float(np.median(ctr))
rep["depth_valid_share"] = float(np.mean(d > 0))
rep["expected_person_front_m"] = 0.8
np.save(a.out / f"{a.body}_depth.npy", d.astype(np.float32))
Image.fromarray((np.clip(d / 3.0, 0, 1) * 255).astype(np.uint8)).resize((320, 240), Image.NEAREST).save(
    a.out / f"{a.body}_depth.png")
rgb = Camera(prim_path=f"{w.robot.eye_prim}/rgb_probe", resolution=(640, 480))
rgb.initialize()
rgb.set_focal_length(1.0)
rgb.set_horizontal_aperture(2 * math.tan(math.radians(c.hfov_deg) / 2))
for _ in range(5):
    w.world.step(render=True)
f = rgb.get_rgba()
if f is not None and np.size(f):
    Image.fromarray(np.asarray(f[..., :3], np.uint8)).save(a.out / f"{a.body}_eye_rgb.png")
# a scripted lunge through MotionPerception, with the robot still
perc = MotionPerception(c, articulated_target=True)
dist = float(np.hypot(person.xy[0] - x, person.xy[1] - y))
perc.end_window(dist - RADIUS + GAP0)
perc.events = []
person.lunges = [Lunge(w.t + 4.0, 0.25 + GAP0, 0.6)]  # quiet for ~2.3 s, then a 0.25 m reach peaking after 4 s
reaches = []
trace = []
frames = []
for k in range(300):
    w.step(drv.idle, drv, False)
    if k % 5 == 0:
        dk = w.depth()
        perc.frame(dk, False, dist - RADIUS + GAP0, w.robot.eye_offset)
        frames.append(dk.astype(np.float32))
        mid = dk[c.height // 4:3 * c.height // 4, c.width // 3:2 * c.width // 3]
        trace.append((round(w.t, 2), round(person.disp, 3), round(float(np.percentile(mid[mid > 0], 5)), 3),
                      perc.event is not None))
rep["lunge_trace (t, displacement, near depth p5, event open)"] = trace
np.savez_compressed(a.out / f"{a.body}_lunge_frames.npz", frames=np.array(frames), trace=np.array(trace, float),
                    d_nav=dist - RADIUS + GAP0, eye_offset=w.robot.eye_offset)
reaches = perc.end_window(dist - RADIUS + GAP0)
rep["lunge_reaches_seen"] = reaches
print(json.dumps(rep, indent=1, default=str), flush=True)
(a.out / f"{a.body}_eye.json").write_text(json.dumps(rep, indent=1, default=str))
from apm_isaac import hard_exit  # noqa: E402

hard_exit(0)
