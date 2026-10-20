"""M2 diagnostic: which headless frame-capture path returns images in this environment?

    $ISAAC_PY experiments/M_release/isaac/m2_render_diag.py --out out/diag

Minimal scene: ground, a lit red cube, a camera.
- A: `isaacsim.sensors.camera.Camera` with `world.step(render=True)`;
- B: the same camera, with extra `simulation_app.update()` calls;
- C: a Replicator render product with an rgb/depth annotator, stepped by `rep.orchestrator.step()`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
launcher = AppLauncher({"headless": True, "enable_cameras": True})
app = launcher.app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for _ext in ("isaacsim.sensors.camera", "isaacsim.sensors.physics", "isaacsim.storage.native"):
    enable_extension(_ext)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Sdf, UsdLux  # noqa: E402

a.out.mkdir(parents=True, exist_ok=True)
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
light = UsdLux.DistantLight.Define(world.stage, Sdf.Path("/World/sun"))
light.CreateIntensityAttr(3000)
world.scene.add(DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([2.0, 0, 0.3]), size=0.5,
                              color=np.array([1.0, 0, 0])))
cam = Camera(prim_path="/World/cam", position=np.array([0.0, 0.0, 0.6]),
             orientation=euler_angles_to_quat(np.array([0, 5, 0]), degrees=True), resolution=(640, 360))
world.reset()
cam.initialize()
cam.add_distance_to_image_plane_to_frame()
for _ in range(20):
    world.step(render=True)
r = cam.get_rgba()
print("A shape", None if r is None else r.shape, flush=True)
for _ in range(20):
    app.update()
r = cam.get_rgba()
print("B shape", None if r is None else r.shape, "depth", np.shape(cam.get_depth()), flush=True)
if r is not None and r.size:
    Image.fromarray(np.asarray(r[..., :3], np.uint8)).save(a.out / "B.png")
rp = rep.create.render_product("/World/cam", (640, 360))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
rgb.attach([rp])
dep.attach([rp])
for _ in range(5):
    rep.orchestrator.step(rt_subframes=4)
x, z = rgb.get_data(), dep.get_data()
print("C shape", np.shape(x), "depth", np.shape(z), "mean", float(np.mean(x)) if np.size(x) else None, flush=True)
if np.size(x):
    Image.fromarray(np.asarray(x[..., :3], np.uint8)).save(a.out / "C.png")
    np.save(a.out / "C_depth.npy", np.asarray(z, np.float32))
app.close()
