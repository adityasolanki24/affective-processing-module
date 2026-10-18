"""M2: make the candidate scenes functional: collision geometry, floor height, free space, renders.

    $ISAAC_PY experiments/M_release/isaac/m2_scenes.py --out out/scenes

For each scene (Simple_Room as the home, the full warehouse, the office, the hospital and the digital-twin
warehouse) it writes:
- `<scene>_occ.npz`: a 5 cm occupancy grid from PhysX raycasts. From 2.5 m down: the floor height is the
  most common hit height, and a cell is blocked if its hit is > 6 cm above the floor. No hit at all means no
  collision geometry there;
- `<scene>_top.png`: a top-down render;
- `<scene>.json`: bounds, floor height, the share of the bounds with colliders and of the floor that is free, and
  the largest free rectangle (candidate interaction area).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
launcher = AppLauncher({"headless": True, "enable_cameras": True})
app = launcher.app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for _ext in ("isaacsim.sensors.camera", "isaacsim.storage.native"):
    enable_extension(_ext)

import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

SCENES = {
    "home": "/Isaac/Environments/Simple_Room/simple_room.usd",
    "warehouse": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "office": "/Isaac/Environments/Office/office.usd",
    "hospital": "/Isaac/Environments/Hospital/hospital.usd",
    "dt_warehouse": "/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
}
root = get_assets_root_path()
a.out.mkdir(parents=True, exist_ok=True)


def largest_free_rect(free: np.ndarray) -> tuple[int, int, int, int]:
    """Largest all-free axis-aligned rectangle (row0, col0, rows, cols) by the histogram method."""
    best = (0, 0, 0, 0, 0)
    h = np.zeros(free.shape[1], int)
    for r in range(free.shape[0]):
        h = np.where(free[r], h + 1, 0)
        stack: list[int] = []
        for c in range(len(h) + 1):
            cur = h[c] if c < len(h) else 0
            while stack and h[stack[-1]] >= cur:
                top = stack.pop()
                width = c if not stack else c - stack[-1] - 1
                area = h[top] * width
                if area > best[0]:
                    best = (area, r - h[top] + 1, (stack[-1] + 1) if stack else 0, h[top], width)
            stack.append(c)
    return best[1:]


for name, rel in SCENES.items():
    res: dict = {"usd": rel}
    try:
        open_stage(root + rel)
        world = World(stage_units_in_meters=1.0)
        world.reset()
        for _ in range(10):
            world.step(render=True)
        stage = omni.usd.get_context().get_stage()
        rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
            stage.GetPseudoRoot()).ComputeAlignedRange()
        lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
        res["bounds"] = [lo.tolist(), hi.tolist()]
        q = omni.physx.get_physx_scene_query_interface()
        step = 0.05
        xs = np.arange(lo[0] + step / 2, hi[0], step)
        ys = np.arange(lo[1] + step / 2, hi[1], step)
        top = min(hi[2], lo[2] + 2.5)
        hz = np.full((len(ys), len(xs)), np.nan)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                hit = q.raycast_closest((float(x), float(y), float(top)), (0.0, 0.0, -1.0), 10.0)
                if hit["hit"]:
                    hz[i, j] = hit["position"][2]
        valid = np.isfinite(hz)
        floor = float(np.round(np.median(hz[valid]), 3)) if valid.any() else None
        free = valid & (hz <= (floor or 0) + 0.06)
        r0, c0, nr, nc = largest_free_rect(free)
        res.update({"floor_z": floor, "collider_coverage": float(valid.mean()), "free_share": float(free.mean()),
                    "largest_free_rect_m": {"x0": float(xs[c0]), "y0": float(ys[r0]), "w": float(nc * step),
                                            "h": float(nr * step)}})
        np.savez_compressed(a.out / f"{name}_occ.npz", hz=hz, xs=xs, ys=ys, floor=floor or 0.0)
        centre = (lo + hi) / 2
        span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        cam = Camera(prim_path="/World/topdown", position=np.array([centre[0], centre[1], lo[2] + 0.9 * span + 3]),
                     orientation=euler_angles_to_quat(np.array([0, 90, 0]), degrees=True), resolution=(960, 960))
        cam.initialize()
        for _ in range(30):
            world.step(render=True)
        img = cam.get_rgba()
        if img is not None and img.size:
            Image.fromarray(np.asarray(img[..., :3], np.uint8)).save(a.out / f"{name}_top.png")
        world.clear_instance()
    except Exception as e:  # noqa: BLE001
        import traceback
        res["error"] = repr(e)
        res["trace"] = traceback.format_exc()[-2000:]
    print(name, {k: v for k, v in res.items() if k != "trace"}, flush=True)
    (a.out / f"{name}.json").write_text(json.dumps(res, indent=1))
app.close()
