"""M4 debug: where are the robot, the person's collider and the person's rendered character? Top-down and side
renders with the collider made visible (red), plus the positions each system reports.

    $ISAAC_PY experiments/M_release/isaac/m4_scene_debug.py --body nova_carter --out out/scene_debug [--no-graph]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--body", default="nova_carter")
ap.add_argument("--no-graph", action="store_true", help="characters without the AnimationGraph")
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac import ANIM_KIT_ARGS, hard_exit  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True, "kit_args": "" if a.no_graph else ANIM_KIT_ARGS}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
import numpy as np  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdShade  # noqa: E402

from apm_isaac.agents import find_character  # noqa: E402
from apm_isaac.world import IsaacWorld, SceneSpec, _look_at  # noqa: E402

a.out.mkdir(parents=True, exist_ok=True)
root = get_assets_root_path()
scene = SceneSpec("grid", "/Isaac/Environments/Grid/default_environment.usd", home=(0.0, 0.0, 0.0),
                  stations={"P1": (1.5, 0.0)})
w = IsaacWorld(scene, a.body, seed=1, characters={"P1": find_character(root, "male_adult_construction_05_new")})
p = w.people["P1"]
cap = UsdGeom.Capsule(w.world.stage.GetPrimAtPath(p.cap_path))  # type: ignore[attr-defined]
cap.CreatePurposeAttr("default")  # make the collider visible for this debug render
mat = UsdShade.Material.Define(w.world.stage, "/World/Looks/red")
sh = UsdShade.Shader.Define(w.world.stage, "/World/Looks/red/s")
sh.CreateIdAttr("UsdPreviewSurface")
from pxr import Sdf  # noqa: E402

sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(mat)
drv = w._driver(0)
for _ in range(60):
    w.step(drv.idle, drv, False)
x, y, yaw = w.robot.pose()
rep = {"robot_physics": [x, y, yaw], "station": p.xy.tolist(),
       "collider_physics": np.asarray(SingleRigidPrim(p.cap_path).get_world_pose()[0]).tolist(),  # type: ignore[attr-defined]
       "character_usd": list(UsdGeom.XformCache().GetLocalToWorldTransform(
           w.world.stage.GetPrimAtPath(p.vis_path)).ExtractTranslation()),  # type: ignore[attr-defined]
       "animated": getattr(p, "animated", None), "eye_pose": [v.tolist() for v in w.eye_pose()]}
cams = {"top": (np.array([0.75, 0.0, 6.0]), np.array([0.75, 0.001, 0.0])),
        "side": (np.array([0.75, -4.0, 1.2]), np.array([0.75, 0.0, 0.6]))}
for name, (eye, tgt) in cams.items():
    c = Camera(prim_path=f"/World/dbg_{name}", resolution=(960, 540))
    c.initialize()
    c.set_focal_length(1.8)
    c.set_world_pose(position=eye, orientation=_look_at(eye, tgt), camera_axes="world")
    for _ in range(8):
        w.world.step(render=True)
    f = c.get_rgba()
    if f is not None and np.size(f):
        tag = "_nograph" if a.no_graph else ""
        Image.fromarray(np.asarray(f[..., :3], np.uint8)).save(a.out / f"{a.body}_{name}{tag}.png")
d = w.depth()
rep["eye_depth_centre"] = float(np.median(d[20:28, 28:36]))
rep["eye_depth_min"] = float(d[d > 0].min()) if np.any(d > 0) else None
Image.fromarray((np.clip(d / 3.0, 0, 1) * 255).astype(np.uint8)).resize((320, 240), Image.NEAREST).save(
    a.out / f"{a.body}_eye_depth{'_nograph' if a.no_graph else ''}.png")
print("debug", json.dumps(rep, default=str), flush=True)
_ = math
hard_exit(0)
