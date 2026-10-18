"""M5/M6: APM lifetimes in Isaac Sim — one integrated runtime (`apm_transfer.runtime.Runtime`) on a physical robot in
a realistic scene, with three people at their stations.

    $ISAAC_PY experiments/M_release/isaac/m5_lifetime.py --robot nova_carter --scene warehouse --experiment M5 \\
        --partition T --first 0 --n 4 --systems RANDINTENT --histories risky_B,safe --out out/m5/T.jsonl
        [--types-model F --edges F --time-model F --task-config F] [--policy-dir DIR] [--video]

**Lifetime** (K's design, shortened). `--n-enc` encounters (default 24). Each encounter meets B1 (probability 0.5),
A1 or C1 (0.25 each), each at its own station. Only B1's behaviour depends on the history; it switches at encounter
`--switch` (default 10):

| history | B1 before | B1 after |
|---|---|---|
| risky_B | pushy | pushy |
| safe | calm | calm |
| change_B | pushy | calm |
| late_B | calm | pushy |

**Systems.** All share the scene, the people, the robot and its safety layers, the seeds (paired: the same lunge
streams), and the runtime's memory class. They differ only in what the runtime is allowed to do:

| system | runtime policy | memory |
|---|---|---|
| DT | decision-theoretic, no VOI (the v1.0 default) | learns from every observation |
| DT_VOI | decision-theoretic with value of information | learns |
| NOMEM | DT | never learns (the kind prior every time) |
| FROZEN | DT | learns until the switch, then frozen |
| SHUFFLED | DT | learns, but each observation is filed under a random identity |
| FIXED | one fixed expression (proximity 0.5, tempo 0.7) | learns (unused by the decision) |
| NEUTRAL | neutral motion at the hand-over distance | learns (unused) |
| LOOKUP | H's lookup, L's tuned parameters | learns |
| MONOTONE | L's tuned monotonic rule (calibrated simple rule) | learns |
| REACTIVE | memory-free; neutral approach, 3 windows, leaves when its detector fires | none |
| RANDINTENT | random grid intent and 1–3 windows (exploration data for fitting) | learns |

APM's risk supervisor applies to the APM systems (DT, DT_VOI, NOMEM, FROZEN, SHUFFLED); every body's own physical
safety layer applies to all.

**Rows** follow `k_run.py` (so `k_fit.py` and the K/L analyses read them), plus the physical stopping distance,
PhysX contact count and impulse, the decision's risk and veto, and wall time. The runtime's JSONL log is kept per
lifetime for replay (`--logs`).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--robot", choices=("nova_carter", "humanoid", "nerva", "wheeled_reference"), required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--experiment", default="M5")
ap.add_argument("--partition", required=True)
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--systems", required=True)
ap.add_argument("--histories", default="change_B,late_B")
ap.add_argument("--n-enc", type=int, default=24)
ap.add_argument("--switch", type=int, default=10)
ap.add_argument("--types-model", default=None)
ap.add_argument("--edges", default=None)
ap.add_argument("--time-model", default=None)
ap.add_argument("--task", default="handoff")
ap.add_argument("--task-config", default=None,
                help="JSON with this robot's task values (retry_s from its own time model)")
ap.add_argument("--policy-dir", default=None)
ap.add_argument("--video", action="store_true")
ap.add_argument("--logs", type=Path, default=None)
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_isaac import ANIM_KIT_ARGS  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True, "kit_args": ANIM_KIT_ARGS}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
import numpy as np  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from apm_isaac.agents import find_character  # noqa: E402
from apm_isaac.body import IsaacBody  # noqa: E402
from apm_isaac.world import IsaacWorld, SceneSpec  # noqa: E402
from apm_transfer.expression_map import GRID, band_of  # noqa: E402
from apm_transfer.infoseek import Task  # noqa: E402
from apm_transfer.runtime import Observation, Runtime  # noqa: E402

M = REPO / "experiments" / "M_release"
L = REPO / "experiments" / "L_audit_perception" / "configs"
K = REPO / "experiments" / "K_recovery" / "configs"
ENTITIES = (("A1", "A", "P1"), ("B1", "B", "P2"), ("C1", "C", "P3"))
P_ENTITY = (0.25, 0.5, 0.25)
CHARACTERS = {"P1": "male_adult_construction_05_new", "P2": "F_Business_02", "P3": "M_Medical_01"}
TUNED = json.loads((L / "LE_tuned.json").read_text())["tuned"]
SYSTEMS = {
    "DT": {"policy": "greedy"}, "DT_VOI": {"policy": "voi"}, "NOMEM": {"policy": "greedy", "memory": "none"},
    "FROZEN": {"policy": "greedy", "memory": "frozen"}, "SHUFFLED": {"policy": "greedy", "memory": "shuffled"},
    "FIXED": {"policy": "fixed:proximity=0.5/tempo=0.7/hesitation=0", "supervise": False},
    "NEUTRAL": {"policy": "neutral", "supervise": False},
    "LOOKUP": {"policy": "lookup:" + TUNED[a.task]["E1"].split(":", 1)[1], "supervise": False},
    "MONOTONE": {"policy": "monotone:" + TUNED[a.task]["E2"].split(":", 1)[1], "supervise": False},
    "REACTIVE": {"policy": "reactive", "supervise": False, "memory": "none"},
    "RANDINTENT": {"policy": "random", "supervise": False},
}


def btype(history: str, k: int) -> str:
    before, after = {"risky_B": ("pushy", "pushy"), "safe": ("calm", "calm"), "change_B": ("pushy", "calm"),
                     "late_B": ("calm", "pushy")}[history]
    return before if k < a.switch else after


def seeds_mod():
    spec = importlib.util.spec_from_file_location("m_seeds", M / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


S = seeds_mod()
types_model = Path(a.types_model) if a.types_model else L / "LC_wheeled_vision_types.json"  # bootstrap: exploration
mcfg = json.loads(types_model.read_text())
if "model" not in mcfg:
    mcfg = {"model": mcfg, "oracle": {}}
edges = json.loads(Path(a.edges).read_text())["acc8"] if a.edges else [0.1 * i for i in range(1, 8)]
time_coef = json.loads(Path(a.time_model or (REPO / "experiments" / "J_validation_transfer" / "configs" /
                                             "J6_model.json")).read_text())["time_coef"]
tcfg = json.loads(Path(a.task_config).read_text()) if a.task_config else json.loads((K / "K_tasks.json").read_text())
task_spec = tcfg["tasks"][a.task]
task = Task(a.task, **{k: v for k, v in task_spec.items() if k != "why"})
voi = tcfg["voi"]

root = get_assets_root_path()
cfg = json.loads((M / "configs" / "scenes.json").read_text())[a.scene]
scene = SceneSpec(a.scene, cfg["usd"], home=tuple(cfg["home"]),
                  stations={k: tuple(v) for k, v in cfg["stations"].items()}, floor_z=cfg["floor_z"],
                  scale=cfg.get("scale", 1.0), held_out=cfg.get("role", "").startswith("held"))
chars = {sid: find_character(root, n) for sid, n in CHARACTERS.items() if sid in scene.stations}
channels = json.loads((REPO / "experiments" / "J_validation_transfer" / "configs" / "H_admitted_interface.json")
                      .read_text())["interface"]
world = IsaacWorld(scene, a.robot, seed=0, video=a.video, characters={k: v for k, v in chars.items() if v},
                   physics_dt=0.005 if a.robot == "humanoid" else 0.002, policy_dir=a.policy_dir,
                   nerva_channels=channels)
stations = {ident: st for ident, _, st in ENTITIES}
a.out.parent.mkdir(parents=True, exist_ok=True)
if a.logs:
    a.logs.mkdir(parents=True, exist_ok=True)
print("robot", json.dumps(getattr(world.robot, "measured", None) or {}, default=str)[:600],
      "animated", {k: getattr(p, "animated", False) for k, p in world.people.items()}, flush=True)

for index in range(a.first, a.first + a.n):
    for history in a.histories.split(","):
        for system in a.systems.split(","):
            spec = SYSTEMS[system]
            base = int(S.world_seed(a.experiment, a.partition, index).generate_state(1)[0])
            rng = np.random.default_rng(base)
            order = [ENTITIES[int(i)] for i in rng.choice(3, a.n_enc, p=P_ENTITY)]
            shuffle_rng = np.random.default_rng(base + 5)
            world.new_lifetime(base + 1)
            log = a.logs / f"{a.robot}_{a.scene}_{system}_{history}_{a.partition}{index}.jsonl" if a.logs else None
            if log is not None and log.exists():
                log.unlink()
            rt = Runtime(mcfg, time_coef, task, voi, lam=0.02, seed=base + 3, camera=True, log=log,
                         policy=spec["policy"], supervise=spec.get("supervise", True))
            body = IsaacBody(world, edges, stations)
            rows, t0 = [], time.time()
            for k, (ident, kind, _) in enumerate(order):
                etype = btype(history, k) if ident == "B1" else "calm"
                b_before = rt.memory.belief(ident, kind)
                d = rt.decide(ident, kind, context=f"{a.scene}:{k}")
                tk = time.time()
                obs, truth = body.encounter(int(base + 100 + k), etype, d)
                out = body.last_out
                mem = spec.get("memory", "learn")
                if mem == "learn" or (mem == "frozen" and k < a.switch):
                    rt.observe(obs)
                elif mem == "shuffled":
                    other = ENTITIES[int(shuffle_rng.integers(3))]
                    rt.observe(Observation(other[0], other[1], obs.band, obs.imu, obs.camera, obs.detected,
                                           obs.context, obs.report))
                d_abs = obs.report.get("d_stop", out["final_distance"])
                imu_peaks = [round(e[1], 5) for e in out["windows"] if e[10] >= 25]
                rows.append({
                    "k": k, "entity": ident, "station": stations[ident], "type": etype,
                    "b_before": [round(float(v), 5) for v in b_before],
                    "b_after": [round(float(v), 5) for v in rt.memory.belief(ident, kind)],
                    "p_before": round(d.p_risk, 5), "u_before": round(d.uncertainty, 4), "action": d.action,
                    "grid_i": d.action if d.action >= 0 else None, "proximity": round(d.proximity, 3),
                    "tempo": round(d.tempo, 3), "hesitation": round(d.hesitation, 3), "windows_planned": d.windows,
                    "windows_done": len(out["windows"]), "aborted": out["aborted"], "standoff": d.standoff,
                    "risk": round(d.predicted_risk, 4), "vetoed": d.vetoed, "reason": d.reason,
                    "imu_cats": list(obs.imu), "imu_peaks": imu_peaks, "cam_codes": list(obs.camera),
                    "cam_reaches": out.get("camera_reaches"), "detected": obs.detected, "band": obs.band,
                    "d_stop": d_abs, "d_stop_physical": obs.report.get("d_stop_physical"),
                    "intended_band": band_of(d.proximity), "contacts": truth["contacts"],
                    "impulse_max": round(truth["contact_impulse_max"], 4), "arrived": out["arrived"],
                    "t_arrive": out["t_arrive"], "trips": out["trips"], "fell": out["fell"],
                    "task_success": task.p_success(d_abs),
                    "reached_interaction": bool(task.d_task is not None and d_abs <= task.d_task + 0.05),
                    "hold_max_dist": out.get("hold_max_dist"),
                    "hold_completed": task.hold_completed(out["arrived"], out.get("hold_max_dist")),
                    "wall_s": round(time.time() - tk, 1)})
                if out["fell"]:
                    break  # a fall ends the lifetime; it is counted
            rec = {"experiment": a.experiment, "partition": a.partition, "index": index, "history": history,
                   "system": system, "robot": a.robot, "scene": a.scene, "held_out": scene.held_out, "task": a.task,
                   "engine": "isaac sim 5.1 / physx", "rows": rows, "wall_s": round(time.time() - t0, 1),
                   "vetoes": rt.safety.vetoes, "types_model": str(types_model), "edges": edges}
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{system} {history} {a.partition}{index}: {len(rows)} enc, falls {sum(r['fell'] for r in rows)}, "
                  f"contacts {sum(r['contacts'] for r in rows)}, wall {rec['wall_s']} s", flush=True)
            if a.video and world.video_frames:
                import imageio
                vid = a.out.parent / f"{a.robot}_{a.scene}_{system}_{history}_{a.partition}{index}.mp4"
                imageio.mimsave(vid, world.video_frames, fps=25)
_ = (GRID, os)
from apm_isaac import hard_exit  # noqa: E402

hard_exit(0)
