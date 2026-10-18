"""M4 smoke test: APM encounters in a realistic Isaac Sim scene (`configs/scenes.json`), one person at each station.

    $ISAAC_PY experiments/M_release/isaac/m4_encounter.py --out out/m4 --body nova_carter --scene warehouse [--video]
        [--policy-dir experiments/M_release/policies/t1_isaaclab_v1]   (humanoid)

It runs one continuous lifetime: each station's person is met in turn, calm then pushy, with fixed intents (close
approach; OBSERVE stand-off for the second pushy meeting).

Reported per encounter:
- what the robot sensed: IMU window peaks, its own detector, camera codes and reaches;
- outcome: arrival, stopping distance, trips, falls;
- truth: PhysX contacts and impulse;
- wall time.

With `--video` it also writes a side-view MP4 of the whole lifetime.
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
ap.add_argument("--body", default="nova_carter", choices=("nova_carter", "humanoid", "nerva", "wheeled_reference"))
ap.add_argument("--scene", default="warehouse")
ap.add_argument("--policy-dir", default=None)
ap.add_argument("--video", action="store_true")
ap.add_argument("--stations", type=int, default=1, help="how many stations to visit")
ap.add_argument("--approach-from", type=float, default=None, help="experiment mode: start each encounter here")
a, _ = ap.parse_known_args()
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac import ANIM_KIT_ARGS  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True, "kit_args": ANIM_KIT_ARGS}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")

from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from apm_isaac.agents import find_character  # noqa: E402
from apm_isaac.world import IsaacWorld, SceneSpec  # noqa: E402
from apm_transfer.expression_map import GRID  # noqa: E402

a.out.mkdir(parents=True, exist_ok=True)
root = get_assets_root_path()
cfg = json.loads((REPO / "experiments" / "M_release" / "configs" / "scenes.json").read_text())[a.scene]
stations = dict(list(cfg["stations"].items())[:a.stations])
names = ["male_adult_construction_05_new", "F_Business_02", "M_Medical_01"]
chars = {sid: find_character(root, names[i % len(names)]) for i, sid in enumerate(stations)}
print("characters", chars, flush=True)
scene = SceneSpec(a.scene, cfg["usd"], home=tuple(cfg["home"]), stations={k: tuple(v) for k, v in stations.items()},
                  floor_z=cfg["floor_z"], scale=cfg.get("scale", 1.0))
dt = 0.005 if a.body == "humanoid" else 0.002
channels = json.loads((REPO / "experiments" / "J_validation_transfer" / "configs" / "H_admitted_interface.json")
                      .read_text())["interface"]
w = IsaacWorld(scene, a.body, seed=7, video=a.video, characters={k: v for k, v in chars.items() if v},
               physics_dt=dt, policy_dir=a.policy_dir, nerva_channels=channels)
print("robot", json.dumps(getattr(w.robot, "measured", None) or {"front": w.robot.front}, default=str)[:2000],
      "animated", {k: getattr(p, "animated", False) for k, p in w.people.items()}, "patrols", w.patrols, flush=True)
plan = [(sid, et, gi, so) for sid in stations for et, gi, so in
        (("calm", 14, 0.0), ("pushy", 14, 0.0), ("pushy", 2, 0.3))]
rows = []
for k, (sid, etype, gi, standoff) in enumerate(plan):
    t0 = time.time()
    out, truth = w.encounter(sid, etype, GRID[gi], windows=2, standoff=standoff, video=a.video,
                             approach_from=a.approach_from, return_home=a.approach_from is None)
    row = {"k": k, "station": sid, "type": etype, "grid": gi, "standoff": standoff, "arrived": out["arrived"],
           "d_stop": out["final_distance"], "imu_peaks": [round(e[1], 4) for e in out["windows"]],
           "detected": [bool(e[9]) for e in out["windows"]], "camera": out["camera"],
           "camera_reaches": out["camera_reaches"], "trips": out["trips"], "fell": out["fell"],
           "contacts": truth["contacts"], "impulse_max": round(truth["contact_impulse_max"], 3),
           "wall_s": round(time.time() - t0, 1), "sim_t": round(w.t, 1)}
    rows.append(row)
    print(row, flush=True)
    if out["fell"]:
        break
(a.out / f"m4_{a.body}_{a.scene}.json").write_text(json.dumps(rows, indent=1))
if a.video and w.video_frames:
    import imageio
    imageio.mimsave(a.out / f"m4_{a.body}_{a.scene}.mp4", w.video_frames, fps=25)
    print("video frames", len(w.video_frames), flush=True)
from apm_isaac import hard_exit  # noqa: E402

hard_exit(0)
