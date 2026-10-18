"""M4 probe: the people AnimationGraph's variables, and whether a walk cycle plays with the character's root set by
script (set_world_transform every frame).

    $ISAAC_PY experiments/M_release/isaac/m4_walk_probe.py --out out/walk
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac import ANIM_KIT_ARGS, hard_exit  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True, "kit_args": ANIM_KIT_ARGS}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
import carb  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Sdf, Usd, UsdLux  # noqa: E402

from apm_isaac.agents import add_person, find_character, setup_animation  # noqa: E402
from apm_isaac.world import _look_at  # noqa: E402

a.out.mkdir(parents=True, exist_ok=True)
root = get_assets_root_path()
world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=0.02)
world.scene.add_default_ground_plane()
st = world.stage
UsdLux.DomeLight.Define(st, Sdf.Path("/World/Dome")).CreateIntensityAttr(1000.0)
graph = setup_animation(st, root)
gp = st.GetPrimAtPath(graph)
print("graph", graph, "attrs", [x.GetName() for x in gp.GetAttributes()][:40], flush=True)
for q in Usd.PrimRange(gp):
    if "ariable" in q.GetTypeName() or "ariable" in q.GetName():
        print("  var prim", q.GetPath(), q.GetTypeName(), [(x.GetName(), x.Get()) for x in q.GetAttributes()][:8],
              flush=True)
p = add_person(st, "W", np.array([0.0, 0.0]), find_character(root, "M_Medical_01"), graph)
cam = Camera(prim_path="/World/Cam", resolution=(640, 480))
cam.initialize()
cam.set_focal_length(1.8)
world.reset()
from apm_isaac.agents import _graph_character, place  # noqa: E402

frames, joints = [], []
for k in range(240):
    world.step(render=True)
    x = 0.01 * max(0, k - 60) * 1.2 * 0.02 / 0.01  # 1.2 m/s after 1.2 s
    place(st, p, np.array([x, 0.0]), 0.0)
    c = _graph_character(st, p.vis_path)  # type: ignore[attr-defined]
    if k == 60 and c is not None:
        for name, val in (("Action", "Walk"), ("Walk", 1.0)):
            try:
                c.set_variable(name, val)
                print("set", name, val, "->", c.get_variable(name), flush=True)
            except Exception as e:  # noqa: BLE001
                print("set_variable", name, repr(e)[:150], flush=True)
    if c is not None and k % 20 == 0:
        pos, rot = carb.Float3(0, 0, 0), carb.Float4(0, 0, 0, 1)
        try:
            c.get_joint_transform("L_Foot", pos, rot)
            joints.append((k, round(pos[0], 3), round(pos[2], 3)))
        except Exception as e:  # noqa: BLE001
            joints.append((k, repr(e)[:60]))
    if k % 30 == 0:
        eye = np.array([x + 0.5, -3.5, 1.2])
        cam.set_world_pose(position=eye, orientation=_look_at(eye, np.array([x, 0.0, 0.9])), camera_axes="world")
        world.render()
        f = cam.get_rgba()
        if f is not None and np.size(f):
            frames.append(np.asarray(f[..., :3], np.uint8))
print("left foot (k, x, z)", joints, flush=True)
if frames:
    Image.fromarray(np.concatenate(frames, axis=1)).save(a.out / "walk_strip.png")
hard_exit(0)
