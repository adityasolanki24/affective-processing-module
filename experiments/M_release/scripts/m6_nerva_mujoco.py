"""M6, NERVA on its validated MuJoCo backend: the same runtime, systems, seeds, lifetime design and row format as the
Isaac runner (`isaac/m5_lifetime.py`). Every record is labelled engine "mujoco 3.3.0".

Why MuJoCo for NERVA's comparative data: in Isaac Sim, NERVA's frozen controller walks, sees and passes its gate
battery, but it fails physical interaction. A light contact from a person's body topples it, and it cannot walk
on the home scene's floor (DEVIATIONS D3). The brief's fallback applies: keep the validated MuJoCo version and label
the backend.

    python experiments/M_release/scripts/m6_nerva_mujoco.py --partition X --first 0 --n 10 \\
        --systems DT,NOMEM,... --histories change_B,late_B --out out/X_nerva_mujoco.jsonl --workers 64

**World** (programme L): `nerva_world.closed_loop_encounter`:
- NERVA's frozen neutral policy, H's admitted channels and SafetyMonitor;
- the abstract entity at the target, with K/L lunge streams (velocity impulses);
- rendered MuJoCo depth vision (`VisionCamera`).

**Models** (programme L, fitted in MuJoCo on L's own partition T, independent of M's data):
- `LC_nerva_vision_types.json`;
- J4 edges;
- J6 time model;
- K's task values (`retry_s` from the J6 time model, K's frozen rule).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
M = REPO / "experiments" / "M_release"
J = REPO / "experiments" / "J_validation_transfer" / "configs"
L = REPO / "experiments" / "L_audit_perception" / "configs"
K = REPO / "experiments" / "K_recovery" / "configs"
ENTITIES = (("A1", "A"), ("B1", "B"), ("C1", "C"))
P_ENTITY = (0.25, 0.5, 0.25)
TUNED = json.loads((L / "LE_tuned.json").read_text())["tuned"]
SYSTEMS = {
    "DT": {"policy": "greedy"}, "DT_VOI": {"policy": "voi"}, "NOMEM": {"policy": "greedy", "memory": "none"},
    "FROZEN": {"policy": "greedy", "memory": "frozen"}, "SHUFFLED": {"policy": "greedy", "memory": "shuffled"},
    "FIXED": {"policy": "fixed:proximity=0.5/tempo=0.7/hesitation=0", "supervise": False},
    "NEUTRAL": {"policy": "neutral", "supervise": False},
    "LOOKUP": {"policy": "lookup:" + TUNED["handoff"]["E1"].split(":", 1)[1], "supervise": False},
    "MONOTONE": {"policy": "monotone:" + TUNED["handoff"]["E2"].split(":", 1)[1], "supervise": False},
    "REACTIVE": {"policy": "reactive", "supervise": False, "memory": "none"},
    "SWAPPED": {"policy": "greedy", "memory": "swapped"}, "POOLED": {"policy": "greedy", "memory": "pooled"},
    "RANDINTENT": {"policy": "random", "supervise": False},
}


def seeds_mod():
    spec = importlib.util.spec_from_file_location("m_seeds", M / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def job(args: tuple) -> dict:
    partition, index, history, system, n_enc, switch, task_name = args[:7]
    experiment, supervision = (args[7], args[8]) if len(args) > 7 else ("M6", "legacy")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from apm_transfer.infoseek import Task
    from apm_transfer.runtime import NervaBody, Observation, Runtime
    from apm_transfer.vision import VisionCamera
    base_name, _, params = system.partition(":")
    spec = dict(SYSTEMS[base_name])
    if params:
        spec["policy"] = spec["policy"].split(":")[0] + ":" + params
    if supervision == "matched":  # every system under the same APM risk supervisor (remediation 5.1)
        spec["supervise"] = True
    S = seeds_mod()
    base = int(S.world_seed(experiment, partition, index).generate_state(1)[0])
    rng = np.random.default_rng(base)
    order = [ENTITIES[int(i)] for i in rng.choice(3, n_enc, p=P_ENTITY)]
    enc_seeds = rng.integers(0, 2**31, n_enc)
    shuffle_rng = np.random.default_rng(base + 5)
    mcfg = json.loads((L / "LC_nerva_vision_types.json").read_text())
    edges = json.loads((J / "J4_edges.json").read_text())["acc8"]
    time_coef = json.loads((J / "J6_model.json").read_text())["time_coef"]
    tcfg = json.loads((K / "K_tasks.json").read_text())
    task = Task(task_name, **{k: v for k, v in tcfg["tasks"][task_name].items() if k != "why"})
    channels = json.loads((J / "H_admitted_interface.json").read_text())["interface"]
    rt = Runtime(mcfg, time_coef, task, tcfg["voi"], lam=0.02, seed=base + 3, camera=True, policy=spec["policy"],
                 supervise=spec.get("supervise", True))
    body = NervaBody(edges, channels, VisionCamera())
    rows, t0 = [], time.time()
    for k, (ident, kind) in enumerate(order):
        before, after = {"risky_B": ("pushy", "pushy"), "safe": ("calm", "calm"), "change_B": ("pushy", "calm"),
                         "late_B": ("calm", "pushy")}[history]
        etype = (before if k < switch else after) if ident == "B1" else "calm"
        from apm_transfer.memory_routing import decide_identity, observe_identity
        mem = spec.get("memory", "learn")
        dec_ident = decide_identity(mem, ident)
        b_before = rt.memory.belief(dec_ident, kind)
        d = rt.decide(dec_ident, kind, context=f"mujoco:{k}")
        if dec_ident != ident:  # the encounter is with the real person; only memory is routed
            from dataclasses import replace as _replace
            d = _replace(d, identity=ident)
        out, truth = body._run(int(enc_seeds[k]), etype, d)
        obs = body._observation(d.identity, d.kind, out, d.standoff)
        if mem == "learn" or (mem == "frozen" and k < switch):
            rt.observe(obs)
        elif mem in ("shuffled", "pooled", "swapped"):
            target = observe_identity(mem, ident, ENTITIES, shuffle_rng)
            tkind = next((e[1] for e in ENTITIES if e[0] == target), kind)
            rt.observe(Observation(target, tkind, obs.band, obs.imu, obs.camera, obs.detected, obs.context,
                                   obs.report))
        d_abs = out["final_distance"]
        rows.append({
            "k": k, "entity": ident, "type": etype, "b_before": [round(float(v), 5) for v in b_before],
            "b_after": [round(float(v), 5) for v in rt.memory.belief(ident, kind)],
            "p_before": round(d.p_risk, 5), "u_before": round(d.uncertainty, 4), "action": d.action,
            "proximity": round(d.proximity, 3), "tempo": round(d.tempo, 3), "hesitation": round(d.hesitation, 3),
            "windows_planned": d.windows, "windows_done": len(out["windows"]), "aborted": out["aborted"],
            "standoff": d.standoff, "risk": round(d.predicted_risk, 4), "vetoed": d.vetoed,
            "imu_cats": list(obs.imu), "imu_peaks": [round(e[1], 5) for e in out["windows"] if e[10] >= 25],
            "cam_codes": list(obs.camera), "detected": obs.detected, "band": obs.band, "d_stop": d_abs,
            "contacts": truth["contacts"], "max_push": truth.get("max_push"), "arrived": out["arrived"],
            "t_arrive": out["t_arrive"], "trips": out["trips"], "fell": out["fell"],
            "task_success": task.p_success(d_abs),
            "hold_completed": task.hold_completed(out["arrived"], out.get("hold_max_dist"))})
        if out["fell"]:
            break
    return {"experiment": experiment, "partition": partition, "supervision": supervision, "index": index,
            "history": history, "system": system,
            "robot": "nerva_mujoco", "scene": "warehouse", "held_out": False, "task": task_name,
            "engine": "mujoco 3.3.0 (NERVA validated backend; abstract target, rendered MuJoCo vision)",
            "rows": rows, "wall_s": round(time.time() - t0, 1), "vetoes": rt.safety.vetoes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--histories", default="change_B,late_B")
    ap.add_argument("--n-enc", type=int, default=24)
    ap.add_argument("--switch", type=int, default=10)
    ap.add_argument("--task", default="handoff")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--experiment", default="M6", help="seed space; R = the remediation's corrected comparison")
    ap.add_argument("--supervision", choices=("legacy", "matched"), default="legacy",
                    help="matched: every system under APM's risk supervisor (remediation 5.1)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.partition == "X" and (a.experiment != "M6" or a.supervision != "legacy"):
        raise ValueError("programme M's locked test partition is not reused for corrected comparisons")
    jobs = [(a.partition, i, h, s, a.n_enc, a.switch, a.task, a.experiment, a.supervision)
            for i in range(a.first, a.first + a.n) for h in a.histories.split(",") for s in a.systems.split(",")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool, open(a.out, "a", encoding="utf-8") as f:
        for rec in pool.imap_unordered(job, jobs):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f'{rec["system"]} {rec["history"]} {rec["partition"]}{rec["index"]}: {len(rec["rows"])} enc, '
                  f'falls {sum(r["fell"] for r in rec["rows"])}, wall {rec["wall_s"]} s', flush=True)


if __name__ == "__main__":
    main()
