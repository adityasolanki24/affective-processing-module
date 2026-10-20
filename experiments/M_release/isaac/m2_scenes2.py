"""M2 (second version): functional checks on the candidate scenes, loaded the way APM loads them (referenced into a
metre stage).

    $ISAAC_PY experiments/M_release/isaac/m2_scenes2.py --out out/scenes2 [--scenes home,office,...]

Per scene:
- **bounds** in the metre stage, which checks unit scaling (centimetre assets must arrive at metre scale);
- **multi-hit downward raycasts** on a 10 cm grid (capped at 400 × 400) give every collider surface per cell:
  - the floor is the most common lowest surface;
  - a cell is an obstacle if any surface lies between floor + 0.06 and floor + 1.6 m (body height);
  - a cell is free if its floor exists and has no obstacle;
- **interior views:** RGB at 1.4 m height from the four free cells with the most clearance around them, looking
  along +x, −x, +y and −y;
- **outputs:** `<scene>.json` (floor, collider coverage, free share, free spots with their clearance),
  `<scene>_occ.npz`, `<scene>_view*.png`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--scenes", default="home,office,hospital,warehouse")
a, _ = ap.parse_known_args()
app = AppLauncher({"headless": True, "enable_cameras": True}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from scipy import ndimage  # noqa: E402

SCENES = {"home": "/Isaac/Environments/Simple_Room/simple_room.usd",
          "office": "/Isaac/Environments/Office/office.usd",
          "hospital": "/Isaac/Environments/Hospital/hospital.usd",
          "warehouse": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"}
root = get_assets_root_path()
a.out.mkdir(parents=True, exist_ok=True)
for name in a.scenes.split(","):
    res: dict = {"usd": SCENES[name]}
    try:
        create_new_stage()
        world = World(stage_units_in_meters=1.0)
        add_reference_to_stage(root + SCENES[name], "/World/Scene")
        world.reset()
        for _ in range(5):
            world.step(render=True)
        stage = world.stage
        rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
            stage.GetPrimAtPath("/World/Scene")).ComputeAlignedRange()
        lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
        res["bounds_m"] = [lo.round(2).tolist(), hi.round(2).tolist()]
        step = max(0.10, float(max(hi[0] - lo[0], hi[1] - lo[1])) / 400)
        xs = np.arange(lo[0] + step / 2, hi[0], step)
        ys = np.arange(lo[1] + step / 2, hi[1], step)
        q = omni.physx.get_physx_scene_query_interface()
        surfaces: list[list[list[float]]] = [[[] for _ in xs] for _ in ys]
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                hits: list[float] = []

                def cb(h, hits=hits):
                    hits.append(float(h.position[2]))
                    return True
                q.raycast_all((float(x), float(y), float(hi[2]) + 0.5), (0.0, 0.0, -1.0), float(hi[2] - lo[2]) + 1.0,
                              cb)
                surfaces[i][j] = hits
        lowest = np.array([[min(h) if h else np.nan for h in row] for row in surfaces])
        valid = np.isfinite(lowest)
        floor = float(np.round(np.nanmedian(lowest), 3)) if valid.any() else None
        obst = np.array([[any((floor or 0) + 0.06 < z < (floor or 0) + 1.6 for z in h) for h in row]
                         for row in surfaces])
        free = valid & ~obst & (np.abs(lowest - (floor or 0)) < 0.05)
        clear = ndimage.distance_transform_edt(free) * step
        res.update({"floor_z": floor, "grid_step": step, "collider_coverage": round(float(valid.mean()), 3),
                    "free_share": round(float(free.mean()), 3), "max_clearance_m": round(float(clear.max()), 2)})
        spots = []
        c2 = clear.copy()
        for _ in range(4):
            i, j = np.unravel_index(np.argmax(c2), c2.shape)
            if c2[i, j] <= 0:
                break
            spots.append({"x": round(float(xs[j]), 2), "y": round(float(ys[i]), 2),
                          "clearance": round(float(c2[i, j]), 2)})
            r = int(max(2.0, c2[i, j]) / step)
            c2[max(0, i - r):i + r, max(0, j - r):j + r] = 0
        res["free_spots"] = spots
        np.savez_compressed(a.out / f"{name}_occ.npz", lowest=lowest, obst=obst, free=free, xs=xs, ys=ys)
        cam = Camera(prim_path="/World/probe_cam", resolution=(960, 540))
        cam.initialize()
        cam.set_focal_length(1.4)
        for k, s in enumerate(spots[:2]):
            for yaw in (0, 90, 180, 270):
                cam.set_world_pose(position=np.array([s["x"], s["y"], (floor or 0) + 1.4]),
                                   orientation=euler_angles_to_quat(np.array([0, 5, yaw]), degrees=True))
                for _ in range(8):
                    world.step(render=True)
                img = cam.get_rgba()
                if img is not None and np.size(img):
                    Image.fromarray(np.asarray(img[..., :3], np.uint8)).save(a.out / f"{name}_view{k}_{yaw}.png")
        world.clear_instance()
    except Exception as e:  # noqa: BLE001
        import traceback
        res["error"] = repr(e)
        res["trace"] = traceback.format_exc()[-2000:]
    print(name, json.dumps({k: v for k, v in res.items() if k != "trace"}), flush=True)
    (a.out / f"{name}.json").write_text(json.dumps(res, indent=1))
    _ = math
app.close()
