"""M4 smoke test: APM encounters in a real Isaac Sim scene. Wheeled robot, warehouse, one person at a station.

    $ISAAC_PY experiments/M_release/isaac/m4_encounter.py --out out/m4 [--video]

It runs four encounters in one continuous lifetime (calm, pushy, pushy, calm) with fixed intents: close
approach, close approach, OBSERVE stand-off, close approach.

Reported per encounter:
- what the robot sensed: IMU window categories, detector, camera codes and reaches;
- outcome: arrival, stopping distance, trips, falls;
- truth: PhysX contacts and impulse;
- wall time.

With `--video`, it also writes a chase-camera MP4 of the whole lifetime.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--video", action="store_true")
a, _ = ap.parse_known_args()
app = AppLauncher({"headless": True, "enable_cameras": True}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from apm_isaac.agents import find_character  # noqa: E402
from apm_isaac.world import IsaacWorld, SceneSpec  # noqa: E402
from apm_transfer.expression_map import GRID  # noqa: E402

a.out.mkdir(parents=True, exist_ok=True)
root = get_assets_root_path()
ch = find_character(root, "male_adult_construction_05_new") or find_character(root, "F_Business_02")
print("character", ch, flush=True)
scene = SceneSpec("warehouse", "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd", home=(0.0, 0.0, 0.0),
                  stations={"P1": (2.4, 0.0)})
w = IsaacWorld(scene, "wheeled", seed=7, video=a.video, characters={"P1": ch} if ch else None)
plan = [("calm", 14, 0.0), ("pushy", 14, 0.0), ("pushy", 2, 0.3), ("calm", 14, 0.0)]
rows = []
for k, (etype, gi, standoff) in enumerate(plan):
    t0 = time.time()
    out, truth = w.encounter("P1", etype, GRID[gi], windows=2, standoff=standoff, video=a.video)
    row = {"k": k, "type": etype, "grid": gi, "standoff": standoff, "arrived": out["arrived"],
           "d_stop": out["final_distance"], "imu_peaks": [round(e[1], 4) for e in out["windows"]],
           "detected": [bool(e[9]) for e in out["windows"]], "camera": out["camera"],
           "camera_reaches": out["camera_reaches"], "trips": out["trips"], "fell": out["fell"],
           "contacts": truth["contacts"], "impulse_max": round(truth["contact_impulse_max"], 3),
           "wall_s": round(time.time() - t0, 1), "sim_t": round(w.t, 1)}
    rows.append(row)
    print(row, flush=True)
(a.out / "m4_encounters.json").write_text(json.dumps(rows, indent=1))
if a.video and w.video_frames:
    import imageio
    imageio.mimsave(a.out / "m4_wheeled_warehouse.mp4", w.video_frames, fps=25)
    print("video frames", len(w.video_frames), flush=True)
app.close()
