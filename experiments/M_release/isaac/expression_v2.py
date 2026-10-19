"""Expression v2 pilot: a physical history shapes APM's experience state, which shapes a later matched task.

    $ISAAC_PY experiments/M_release/isaac/expression_v2.py --robot humanoid|nerva --history pushy|calm \
        --out out/expr_v2 <model args from cloud/m6_plan.model_args(robot)>

1. **Calibration (label-free).** The robot stands alone at its approach point with nobody within reach, for 20 s.
   Its own peak standing sway per second sets the disturbance threshold (`calibrate_threshold`).
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
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--robot", choices=("humanoid", "nerva"), required=True)
ap.add_argument("--history", choices=("pushy", "calm"), required=True)
ap.add_argument("--scene", default="warehouse")
ap.add_argument("--n-history", type=int, default=8)
ap.add_argument("--n-present", type=int, default=2)
ap.add_argument("--conditions", default="memory,neutral")
ap.add_argument("--unmatched", action="store_true", help="let speed and stopping distance vary too")
ap.add_argument("--seed", type=int, default=73001)
ap.add_argument("--types-model", required=True)
ap.add_argument("--edges", required=True)
ap.add_argument("--time-model", required=True)
ap.add_argument("--task-config", required=True)
ap.add_argument("--policy-dir", default=None)
ap.add_argument("--out", type=Path, required=True)
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
from apm_isaac.robots import DT  # noqa: E402
from apm_isaac.world import IsaacWorld, SceneSpec  # noqa: E402
from apm_transfer.experience_state import ExperienceState, calibrate_threshold  # noqa: E402
from apm_transfer.infoseek import Task  # noqa: E402
from apm_transfer.runtime import Runtime  # noqa: E402

M = REPO / "experiments" / "M_release"
APPROACH_M = {"humanoid": 2.2, "nerva": 1.1}
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
mcfg = json.loads(Path(a.types_model).read_text())
if "model" not in mcfg:
    mcfg = {"model": mcfg, "oracle": {}}
edges = json.loads(Path(a.edges).read_text())["acc8"]
time_coef = json.loads(Path(a.time_model).read_text())["time_coef"]
tcfg = json.loads(Path(a.task_config).read_text())
task = Task("handoff", **{k: v for k, v in tcfg["tasks"]["handoff"].items() if k != "why"})
a.out.mkdir(parents=True, exist_ok=True)
stem = f"{a.robot}_{a.scene}_{a.history}_{a.seed}"


def calibrate(seconds: float = 20.0) -> tuple[float, list[float]]:
    """Stand alone at the approach point (the nearest person 2.2 m / 1.1 m away, out of reach)."""
    sx, sy = world.people[STATIONS["B1"]].xy
    hx, hy, _ = world._home
    dv = np.array([hx - sx, hy - sy])
    dv /= np.linalg.norm(dv) or 1.0
    r = APPROACH_M[a.robot]
    world.robot.teleport(sx + r * dv[0], sy + r * dv[1], math.atan2(-dv[1], -dv[0]))
    from apm_express.adapter import NEUTRAL_REQUEST
    drv = world._driver(a.seed)
    idle = NEUTRAL_REQUEST if a.robot == "nerva" else drv.idle
    pts, peaks = [], []
    for _ in range(int(seconds / DT)):
        world.step(idle, drv)
        pts.append(world.robot.pose()[:2])
    sp = np.r_[0.0, np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1) / DT]
    sm = np.convolve(sp, np.ones(5) / 5, mode="same")
    per = int(round(1.0 / DT))
    for j in range(1, len(sm) // per):  # skip the first second (settling after the teleport)
        peaks.append(float(sm[j * per:(j + 1) * per].max()))
    return calibrate_threshold(peaks), peaks


world.new_lifetime(a.seed)
threshold, calib_peaks = calibrate()
rt = Runtime(mcfg, time_coef, task, tcfg["voi"], lam=0.02, seed=a.seed + 3, camera=True, policy="greedy",
             supervise=True)
rt.experience = ExperienceState()
rt.experience.disturbance.threshold_m_s = threshold
body = IsaacBody(world, edges, STATIONS, video=False, approach_from=APPROACH_M[a.robot])
body.program_mode, body.program_matched = "memory", False
body.experience_source = rt.experience.readout
record: dict = {"robot": a.robot, "scene": a.scene, "history": a.history, "seed": a.seed,
                "engine": "isaac sim 5.1 / physx", "controller": "booster_gym candidate" if candidate else "default",
                "threshold_m_s": threshold, "calibration_peaks": calib_peaks, "history_rows": [], "present": {}}
print("CALIBRATED", a.robot, round(threshold, 4), flush=True)

# 2. history: memory and physics continue; each encounter starts at the approach point (repositioned)
etype = "pushy" if a.history == "pushy" else "calm"
for k in range(a.n_history):
    before = rt.experience.readout("B1")
    d = rt.decide("B1", "B", context=f"history:{k}")
    obs, truth = body.encounter(a.seed + 100 + k, etype, d)
    rt.observe(obs)
    after = rt.experience.readout("B1")
    record["history_rows"].append({
        "k": k, "readout_before": before, "readout_after": after, "self_disp_peaks": obs.report["self_disp_peaks"],
        "camera_reaches": obs.report.get("camera_reaches"), "contacts_truth": truth["contacts"],
        "fell": bool(body.last_out["fell"]), "expression": obs.report.get("expression")})
    print("HISTORY", k, etype, "wariness", round(after["wariness"], 3), "disp", obs.report["self_disp_peaks"],
          "contacts(truth)", truth["contacts"], flush=True)
    if body.last_out["fell"]:
        break
snapshot = a.out / f"{stem}_state.json"
rt.save(snapshot)
record["learned_state"] = rt.experience.readout("B1")
print("LEARNED", json.dumps(record["learned_state"]), flush=True)

# 3. present: identical seeds for every history and condition; B1 behaves calmly; speed/distance matched
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
