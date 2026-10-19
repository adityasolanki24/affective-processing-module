"""Expression v2 pilot: a physical history shapes APM's experience state, which shapes a later matched task.

    $ISAAC_PY experiments/M_release/isaac/expression_v2.py --robot humanoid|nerva --history pushy|calm \
        --out out/expr_v2 <model args from cloud/m6_plan.model_args(robot)>

1. **Calibration (a quiet period, label-free).** Six encounters with a person who does not move, run through
   the same encounter procedure, without memory updates, cycling the neutral, guarded and open postures. The
   robot's own readings set both thresholds:
   - disturbance: its peak unintended base speed in the sensing windows (`calibrate_threshold`);
   - camera: 1.5 x the 95th percentile of its looming peaks (how far surfaces anywhere in view came closer
     within 0.5 s), and at least 0.05 m.
2. **History.** `--n-history` meetings with person B1, whose physical behaviour is pushy or calm. Encounters are
   repositioned (the robot starts each one at its approach point); memory is continuous. The robot already
   expresses its current state during the history. Each meeting updates the runtime's experience state from
   what the robot sensed: its own unintended displacement while standing, and the camera's motion estimates.
3. **Present task.** The learned state is snapshotted. Then `--n-present` meetings with B1 behaving calmly, with
   identical physics seeds for every history and condition, and speed and stopping distance matched:
   - memory: the plan comes from the learned state;
   - neutral: the neutral program (no expression).
   The side-view video has no overlays.

Nothing here reads the person's type or contact truth to choose behaviour. Truth is recorded for evaluation only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--robot", choices=("humanoid", "nerva", "nova_carter"), required=True)
ap.add_argument("--history", choices=("pushy", "calm"), required=True)
ap.add_argument("--scene", default="warehouse")
ap.add_argument("--n-history", type=int, default=8)
ap.add_argument("--n-present", type=int, default=2)
ap.add_argument("--conditions", default="memory,neutral")
ap.add_argument("--unmatched", action="store_true", help="let speed and stopping distance vary too")
ap.add_argument("--seed", type=int, default=73001)
ap.add_argument("--near-miss", type=float, default=None,
                help="lunges stop this far (m) short of the robot: sudden approaches without contact")
ap.add_argument("--looming-near", type=float, default=None,
                help="looming from cells nearer than this (m) only (NERVA: its sway flips silhouette edges)")
ap.add_argument("--looming-upright", action="store_true",
                help="looming from upright surfaces only (NERVA: its low, swaying camera sees the floor's depth swing)")
ap.add_argument("--looming-horizon", type=float, default=0.5,
                help="looming time window (s); NERVA's near-miss lunges (<= 0.15 m/s, stopping short) need 1 s")
ap.add_argument("--types-model", required=True)
ap.add_argument("--edges", required=True)
ap.add_argument("--time-model", required=True)
ap.add_argument("--task-config", required=True)
ap.add_argument("--policy-dir", default=None)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--demo", action="store_true", help="one continuous lifetime (no teleports): B1 pushy for "
                "--pushy-n meetings, then calm for --calm-n; whole lifetime filmed with the learned state per frame")
ap.add_argument("--demo-reposition", action="store_true",
                help="demo: start each meeting at the approach point, as the histories do (no long walk from home; "
                     "T1's v1 policy is trained for turn rates up to 1 rad/s), so meetings are cut together")
ap.add_argument("--pushy-n", type=int, default=6)
ap.add_argument("--calm-n", type=int, default=6)
a, _ = ap.parse_known_args()
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac import ANIM_KIT_ARGS  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True, "kit_args": ANIM_KIT_ARGS}).app
from apm_isaac import exit_on_error, hard_exit  # noqa: E402

exit_on_error()
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
import imageio  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from apm_isaac.agents import find_character  # noqa: E402
from apm_isaac.body import IsaacBody  # noqa: E402
from apm_isaac.world import IsaacWorld, SceneSpec  # noqa: E402
from apm_transfer.experience_state import ExperienceState, calibrate_threshold  # noqa: E402
from apm_transfer.infoseek import Task  # noqa: E402
from apm_transfer.runtime import Runtime  # noqa: E402

M = REPO / "experiments" / "M_release"
APPROACH_M = {"humanoid": 2.2, "nerva": 1.1, "nova_carter": 2.2}
CHARACTERS = {"P1": "male_adult_construction_05_new", "P2": "F_Business_02", "P3": "M_Medical_01"}
STATIONS = {"A1": "P1", "B1": "P2", "C1": "P3"}

root = get_assets_root_path()
cfg = json.loads((M / "configs" / "scenes.json").read_text())[a.scene]
scene = SceneSpec(a.scene, cfg["usd"], home=tuple(cfg["home"]),
                  stations={k: tuple(v) for k, v in cfg["stations"].items()}, floor_z=cfg["floor_z"],
                  scale=cfg.get("scale", 1.0), held_out=cfg.get("role", "").startswith("held"))
chars = {sid: find_character(root, n) for sid, n in CHARACTERS.items()}
channels = json.loads((REPO / "experiments" / "J_validation_transfer" / "configs" / "H_admitted_interface.json")
                      .read_text())["interface"]
candidate = bool(a.policy_dir and json.loads((Path(a.policy_dir) / "env_cfg.json").read_text())
                 .get("controller_kind") == "booster_gym_t1")
world = IsaacWorld(scene, a.robot, seed=a.seed, video=True, characters={k: v for k, v in chars.items() if v},
                   physics_dt=0.005 if a.robot == "humanoid" and not candidate else 0.002, policy_dir=a.policy_dir,
                   nerva_channels=channels, walkers=0)
# Only B1 takes part in these runs: the other two station people are moved out of the scene (they would only
# occlude the review camera; nothing in the experiment involves them).
for sid in ("P1", "P3"):
    world.people[sid].xy = np.array(world.people[sid].xy) + np.array([60.0, 60.0])
mcfg = json.loads(Path(a.types_model).read_text())
if "model" not in mcfg:
    mcfg = {"model": mcfg, "oracle": {}}
edges = json.loads(Path(a.edges).read_text())["acc8"]
time_coef = json.loads(Path(a.time_model).read_text())["time_coef"]
tcfg = json.loads(Path(a.task_config).read_text())
task = Task("handoff", **{k: v for k, v in tcfg["tasks"]["handoff"].items() if k != "why"})
a.out.mkdir(parents=True, exist_ok=True)
stem = f"{a.robot}_{a.scene}_{a.history}_{a.seed}"


def calibrate(n: int = 6) -> tuple[float, float, list[float], list[float]]:
    """Quiet-period calibration: the person stands still; the same encounter procedure; no memory update. The
    encounters cycle the neutral, a fully guarded and a fully open plan, so the sway and camera behaviour of every
    posture the robot can adopt set the thresholds (a guarded posture's own sway must not read as a push)."""
    from apm_express.intent import ExpressionIntent
    from apm_transfer.expressive_program import plan
    postures = [None, {"wariness": 1.0, "uncertainty": 0.0, "familiarity": 1.0},
                {"wariness": 0.0, "uncertainty": 0.0, "familiarity": 1.0}]
    disp, looms = [], []
    for k in range(n):
        out, _ = world.encounter(STATIONS["B1"], "still", ExpressionIntent(proximity=0.75).sanitized(), 3, 0.0,
                                 approach_from=APPROACH_M[a.robot], return_home=False,
                                 program=plan(postures[k % 3], matched=True))
        disp += list(out["self_disp_peaks"])  # the same measurement as at run time (sensing windows only)
        looms += [float(x) for x in out["looming_peaks"]]
        print("CALIBRATION", k, "disp", out["self_disp_peaks"], "looming", out["looming_peaks"],
              "reaches", out["camera_reaches"], flush=True)
    # camera evidence: looming (the nearest central depth dropping within 0.5 s), above the quiet-period noise
    cam = max(0.05, 1.5 * float(np.quantile(looms, 0.95))) if looms else 0.10
    return calibrate_threshold(disp), cam, disp, looms


world.near_miss_margin = a.near_miss
world.looming_near_m = a.looming_near
world.looming_horizon_s = a.looming_horizon
world.looming_upright = a.looming_upright
world.new_lifetime(a.seed)
threshold, cam_threshold, calib_peaks, calib_reaches = calibrate()
rt = Runtime(mcfg, time_coef, task, tcfg["voi"], lam=0.02, seed=a.seed + 3, camera=True, policy="greedy",
             supervise=True)
rt.experience = ExperienceState()
rt.experience.disturbance.threshold_m_s = threshold
rt.experience.movement.minimum_reach_m = cam_threshold
body = IsaacBody(world, edges, STATIONS, video=False, approach_from=APPROACH_M[a.robot])
body.program_mode, body.program_matched = "memory", False
body.experience_source = rt.experience.readout
record: dict = {"robot": a.robot, "scene": a.scene, "history": a.history, "seed": a.seed,
                "engine": "isaac sim 5.1 / physx", "controller": "booster_gym candidate" if candidate else "default",
                "threshold_m_s": threshold, "camera_threshold_m": cam_threshold, "near_miss_m": a.near_miss,
                "looming_near_m": a.looming_near, "looming_horizon_s": a.looming_horizon,
                "looming_upright": a.looming_upright,
                "calibration_peaks": calib_peaks, "calibration_reaches": calib_reaches, "history_rows": [],
                "present": {}}
print("CALIBRATED", a.robot, "disturbance", round(threshold, 4), "camera", round(cam_threshold, 3), flush=True)

# 2. history: memory and physics continue; each encounter starts at the approach point (repositioned)
etype = "pushy" if a.history == "pushy" else "calm"
for k in range(0 if a.demo else a.n_history):
    before = rt.experience.readout("B1")
    d = rt.decide("B1", "B", context=f"history:{k}")
    obs, truth = body.encounter(a.seed + 100 + k, etype, d)
    rt.observe(obs)
    after = rt.experience.readout("B1")
    record["history_rows"].append({
        "k": k, "readout_before": before, "readout_after": after, "self_disp_peaks": obs.report["self_disp_peaks"],
        "camera_reaches": obs.report.get("camera_reaches"), "looming_peaks": obs.report.get("looming_peaks"),
        "contacts_truth": truth["contacts"],
        "fell": bool(body.last_out["fell"]), "expression": obs.report.get("expression")})
    print("HISTORY", k, etype, "wariness", round(after["wariness"], 3), "loom", obs.report.get("looming_peaks"),
          "disp", obs.report["self_disp_peaks"],
          "contacts(truth)", truth["contacts"], flush=True)
    if body.last_out["fell"]:
        break
snapshot = a.out / f"{stem}_state.json"
rt.save(snapshot)
record["learned_state"] = rt.experience.readout("B1")
print("LEARNED", json.dumps(record["learned_state"]), flush=True)

# 3. present: identical seeds for every history and condition; B1 behaves calmly; speed/distance matched.
# One review camera for every condition and history: side-on to the approach line, full body, the less
# occluded side (rendered depth check), at fixed world coordinates.


def review_camera() -> tuple[np.ndarray, np.ndarray]:
    s = np.array(world.people[STATIONS["B1"]].xy, float)
    hx, hy, _ = world._home
    dv = np.array([hx, hy]) - s
    dv /= np.linalg.norm(dv) or 1.0
    r = APPROACH_M[a.robot]
    mid2 = s + 0.5 * r * dv
    side = np.array([-dv[1], dv[0]])
    dist, h, z = {"nerva": (1.15, 0.45, 0.22), "nova_carter": (2.6, 1.2, 0.4)}.get(a.robot, (2.6, 1.3, 0.6))
    look = np.array([*mid2, z])
    eyes = [np.array([*(mid2 + sgn * dist * side), h]) for sgn in (1.0, -1.0)]
    occ = [world._occlusion(e, look) for e in eyes]
    world._video_cam.set_focal_length(2.2)
    return eyes[int(np.argmin(occ))], look


world.video_fixed = review_camera()
record["review_camera"] = {"eye": world.video_fixed[0].tolist(), "look": world.video_fixed[1].tolist()}
if a.demo:  # the complete learning-and-recovery demonstration: continuous physics, the robot walks home each time
    from apm_transfer.expressive_program import plan as _plan
    world.new_lifetime(a.seed + 11)
    body.approach_from, body.video = (APPROACH_M[a.robot] if a.demo_reposition else None), True
    body.program_mode, body.program_matched = "memory", False
    world.video_writer = imageio.get_writer(str(a.out / f"{stem}_demo.mp4"), fps=25, codec="libx264",
                                            quality=8, macro_block_size=8)
    rows = []
    for k in range(a.pushy_n + a.calm_n):
        et = "pushy" if k < a.pushy_n else "calm"
        st = rt.experience.readout("B1")
        pl = _plan(st)
        world.overlay = {"k": k, "person": et, "wariness": round(st["wariness"], 3),
                         "disturbance_p": round(st["disturbance_p"], 3), "movement_p": round(st["movement_p"], 3),
                         "familiarity": round(st["familiarity"], 3), "observe_s": round(pl.observe_s, 2),
                         "guard": round(pl.guard, 2), "openness": round(pl.openness, 2),
                         "retreat_m": round(pl.retreat_m, 2), "directness": round(pl.directness, 2)}
        d = rt.decide("B1", "B", context=f"demo:{k}")
        obs, truth = body.encounter(a.seed + 900 + k, et, d)
        rt.observe(obs)
        out = body.last_out
        rows.append({"k": k, "person": et, "readout_before": st, "readout_after": rt.experience.readout("B1"),
                     "self_disp_peaks": out.get("self_disp_peaks"), "looming_peaks": out.get("looming_peaks"),
                     "expression": out.get("expression"), "motion": out.get("motion"),
                     "fell": bool(out["fell"]), "contacts_truth": truth["contacts"]})
        print("DEMO", k, et, "wariness", round(rt.experience.readout("B1")["wariness"], 3), "fell", out["fell"],
              flush=True)
        if out["fell"]:
            break
    world.video_writer.close()
    world.video_writer = None
    (a.out / f"{stem}_demo_meta.json").write_text(json.dumps({"frames": world.video_meta}, default=str))
    record["demo"] = {"pushy_n": a.pushy_n, "calm_n": a.calm_n, "continuous_physics": not a.demo_reposition,
                      "repositioned_meetings": a.demo_reposition, "rows": rows}
    (a.out / f"{stem}_demo.json").write_text(json.dumps(record, indent=1, default=str))
    print("DEMO DONE", stem, flush=True)
    hard_exit(0)
for cond in a.conditions.split(","):
    rt.load(snapshot)
    body.program_mode, body.program_matched, body.video = cond, not a.unmatched, True
    body.experience_source = rt.experience.readout  # the reloaded experience object
    world.new_lifetime(a.seed + 7)
    world.video_writer = imageio.get_writer(str(a.out / f"{stem}_{cond}.mp4"), fps=25, codec="libx264",
                                            quality=8, macro_block_size=8)
    rows = []
    for k in range(a.n_present):
        d = rt.decide("B1", "B", context=f"present:{k}")
        obs, truth = body.encounter(a.seed + 500 + k, "calm", d)
        rt.observe(obs)
        out = body.last_out
        rows.append({"k": k, "readout": out.get("experience_readout"), "expression": out.get("expression"),
                     "motion": out.get("motion"), "measured_distance": out.get("measured_distance"),
                     "t_arrive": out.get("t_arrive"), "fell": bool(out["fell"]), "trips": out["trips"],
                     "contacts_truth": truth["contacts"]})
        print("PRESENT", cond, k, "fell" if out["fell"] else "ok", flush=True)
        if out["fell"]:
            break
    world.video_writer.close()
    world.video_writer = None
    (a.out / f"{stem}_{cond}_meta.json").write_text(json.dumps({"frames": world.video_meta}, default=str))
    record["present"][cond] = rows
(a.out / f"{stem}.json").write_text(json.dumps(record, indent=1, default=str))
print("EXPRESSION DONE", stem, flush=True)
hard_exit(0)
