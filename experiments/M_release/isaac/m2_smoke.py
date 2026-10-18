"""M2 smoke test: the pinned Isaac Sim 5.1 environment, headless, on the cloud GPU.

    $ISAAC_PY experiments/M_release/isaac/m2_smoke.py --out out/m2

It writes:
- `assets.json`: the stock asset tree under Environments, People and Robots (two levels deep);
- for each candidate scene: whether it loads, its stage bounds, and an RGB + depth frame from a camera
  (`<scene>_rgb.png`, `<scene>_depth.npy`);
- `physics.json`: step rate with and without rendering, and a falling-cube check (gravity, contact rest height).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
app = SimulationApp({"headless": True, "width": 960, "height": 540})

import numpy as np  # noqa: E402
import omni.client  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

out = a.out
out.mkdir(parents=True, exist_ok=True)
root = get_assets_root_path()
report: dict = {"assets_root": root, "env": {k: os.environ.get(k) for k in ("ISAAC_PY",)}}


def ls(path: str, depth: int) -> dict:
    res, entries = omni.client.list(path)
    if res != omni.client.Result.OK:
        return {"_error": str(res)}
    tree = {}
    for e in entries:
        if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN and depth > 0:
            tree[e.relative_path] = ls(f"{path}/{e.relative_path}", depth - 1)
        else:
            tree[e.relative_path] = e.size
    return tree


assets = {d: ls(f"{root}/Isaac/{d}", 2) for d in ("Environments", "People", "Robots")}
(out / "assets.json").write_text(json.dumps(assets, indent=1))
print("asset dirs:", {d: list(t)[:40] for d, t in assets.items()}, flush=True)

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

SCENES = {
    "warehouse": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "office": "/Isaac/Environments/Office/office.usd",
    "hospital": "/Isaac/Environments/Hospital/hospital.usd",
    "simple_room": "/Isaac/Environments/Simple_Room/simple_room.usd",
}
scenes = {}
for name, rel in SCENES.items():
    t0 = time.time()
    try:
        ok = open_stage(root + rel)
        stage = omni.usd.get_context().get_stage()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
            stage.GetPseudoRoot()).ComputeAlignedRange()
        lo, hi = np.array(bbox.GetMin()), np.array(bbox.GetMax())
        world = World(stage_units_in_meters=1.0)
        centre = (lo + hi) / 2
        cam = Camera(prim_path="/World/smoke_cam", position=np.array([centre[0], centre[1], 1.6]),
                     orientation=euler_angles_to_quat(np.array([0, 10, 0]), degrees=True), resolution=(640, 360))
        world.reset()
        cam.initialize()
        cam.add_distance_to_image_plane_to_frame()
        for _ in range(60):
            world.step(render=True)
        rgba = cam.get_rgba()
        depth = cam.get_depth()
        Image.fromarray(np.asarray(rgba[..., :3], np.uint8)).save(out / f"{name}_rgb.png")
        np.save(out / f"{name}_depth.npy", np.asarray(depth, np.float32))
        n_prims = sum(1 for _ in stage.Traverse())
        scenes[name] = {"loaded": bool(ok), "bounds_min": lo.tolist(), "bounds_max": hi.tolist(), "prims": n_prims,
                        "rgb_mean": float(np.mean(rgba[..., :3])) if rgba is not None and rgba.size else None,
                        "depth_valid": float(np.isfinite(depth).mean()) if depth is not None else None,
                        "seconds": round(time.time() - t0, 1)}
        world.clear_instance()
    except Exception as e:  # noqa: BLE001
        scenes[name] = {"error": repr(e), "trace": traceback.format_exc()[-1500:]}
    print(name, scenes[name], flush=True)
report["scenes"] = scenes

# physics: an empty stage with a ground plane and a falling cube
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage  # noqa: E402

create_new_stage()
world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=1 / 50)
world.scene.add_default_ground_plane()
cube = world.scene.add(DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0, 0, 1.0]),
                                     size=0.2, color=np.array([1, 0, 0])))
world.reset()
zs = []
for _ in range(400):
    world.step(render=False)
    zs.append(float(cube.get_world_pose()[0][2]))
t0 = time.time()
for _ in range(2000):
    world.step(render=False)
no_render = 2000 / (time.time() - t0)
t0 = time.time()
for _ in range(200):
    world.step(render=True)
with_render = 200 / (time.time() - t0)
report["physics"] = {"rest_height": zs[-1], "expected": 0.1, "fall_time_steps": int(np.argmax(np.array(zs) < 0.11)),
                     "steps_per_s_no_render": round(no_render, 1), "steps_per_s_render": round(with_render, 1)}
print("physics", report["physics"], flush=True)
(out / "report.json").write_text(json.dumps(report, indent=1))
app.close()
