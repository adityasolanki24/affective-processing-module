"""G7 lifelong learning (prereg/G7_ADDENDUM.md): updating A0's consequence predictor across changing environments.

Phases (each with its own experience, validation and frozen test lifetimes, G7 seeds):
  PH1 nominal · PH2 faster (U(0.8, 1.6) m/s) and more careless (60%) pedestrians · PH3 slippery (3–6 patches) and
  crowded (4–6 of 8 people).
Learning systems (all start from programme F's frozen predictor, A0's rule and memory unchanged):
  L0 frozen      — never updated
  L1 periodic    — after each phase, retrain on base ∪ all experience and deploy (no gate)
  L2 gated       — the same candidate, promoted only if on the phase's validation lifetimes mean J ≥ current and the
                   collision rate ≤ 1.05 × current (paired); otherwise the current predictor stays (rollback)
  L2c corrupted  — as L2, but 30% of the new experience has flipped collision labels (false outcomes)
  L3 no replay   — retrain on the latest phase's experience only, deploy (no gate): forgetting reference
Base data: exploration lifetimes G7-T 0–149 (F's randomised safe action variation). Experience: 60 closed-loop
lifetimes per phase run with the system's *currently deployed* predictor (G7-T 1000·phase + …). Predictors: 3-seed
MLP ensembles (F's architecture and training). Evaluation after each phase's update: frozen suites G7-X of the
current and of the first phase (N lifetimes each, 5 episodes).

    python scripts/g7_lifelong.py --partition X --n 150 --workers 90
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import seeds as S  # noqa: E402

from apm_research.dataset import DecisionRecorder  # noqa: E402
from apm_research.features import WIDTH  # noqa: E402
from apm_research.harness import Lifetime, run_lifetime  # noqa: E402
from apm_research.policies import Explore  # noqa: E402
from apm_research.predictor import ActionPredictor, EnsemblePredictor  # noqa: E402
from apm_select.provenance import confirmatory_lock, environment, write_json  # noqa: E402
from apm_select.regulators import Regulated, make_regulator  # noqa: E402
from apm_testbeds.wheeled_ctrl import WorldConfig, child  # noqa: E402

EPISODES = 5
PHASES = {"PH1": {}, "PH2": {"world": {"ped_speed": (0.8, 1.6)}, "p_careless": 0.6},
          "PH3": {"world": {"patches": (3, 6), "people": (4, 6)}, "n_people": 8}}
SYSTEMS = ("L0", "L1", "L2", "L2c", "L3")
EXPERIENCE, VALIDATION, SEEDS, CORRUPT, BASE = 60, 20, (0, 1, 2), 0.3, 150
_P: dict = {}


def lifetime(phase: str, partition: str, index: int) -> Lifetime:
    c = PHASES[phase]
    cfg = dataclasses.replace(WorldConfig(), **c.get("world", {}))
    return Lifetime(S.world_seed("G7", partition, index), n_episodes=EPISODES, config=cfg,
                    n_people=c.get("n_people", 6), p_careless=c.get("p_careless", 0.4))


def a0(pred) -> Regulated:
    g1 = json.loads((HERE / "configs" / "G1_selected.json").read_text())
    s = g1["systems"][g1["best_simple"]]
    return Regulated(pred, s["lam_c"], s["lam_s"], make_regulator(s["regulator"], s.get("params", {})))


def _predictor(d: dict):
    key = json.dumps(d["members"][0]["weights"][0][1][:3])  # cheap cache key
    if key not in _P:
        _P[key] = EnsemblePredictor.from_dict(d)
    return _P[key]


def run_job(job: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    lt = lifetime(job["phase"], job["partition"], job["index"])
    if job["kind"] == "explore":
        rec = DecisionRecorder()
        run_lifetime(lt, Explore(child(S.world_seed("G7", "T", job["index"]), 500)), recorder=rec)
        return {"rows": rec.rows}
    pred = _predictor(job["predictor"])
    rec = DecisionRecorder() if job["kind"] == "experience" else None
    r = run_lifetime(lt, a0(pred), recorder=rec)
    out = {"system": job.get("system"), "phase": job["phase"], "suite": job.get("suite"), "index": job["index"],
           "totals": {k: float(r[k]) for k in ("J", "collisions", "minutes", "goals_completed", "goals_issued",
                                                "severe", "idle", "periods")}}
    if rec is not None:
        out["rows"] = rec.rows
    return out


def arrays(rows: list[dict]):
    rec = DecisionRecorder()
    rec.rows = rows
    return rec.arrays()


def fit(args):
    seed, train, val = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    return ActionPredictor(WIDTH, seed=seed).fit(*arrays(train), *arrays(val))


def train(pool, rows: list[dict], rng: np.random.Generator) -> dict:
    idx = rng.permutation(len(rows))
    cut = int(0.85 * len(rows))
    tr, va = [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]
    members = pool.map(fit, [(s, tr, va) for s in SEEDS])
    return EnsemblePredictor(members).to_dict()


def corrupt(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    out = []
    for r in rows:
        if rng.random() < CORRUPT:
            r = {**r, "y_col": 1.0 - r["y_col"]}
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G7"))
    args = ap.parse_args()
    provenance = confirmatory_lock(HERE / "prereg" / "G7_ADDENDUM.md") if args.partition in ("X", "S") \
        else environment()
    start = time.time()
    rng = np.random.default_rng(20261012)
    base_pred = json.loads(Path("runs/E2/predictor.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    log, evals = [], []
    with Pool(args.workers) as pool:
        base = [r for out in pool.map(run_job, [{"kind": "explore", "phase": "PH1", "partition": "T", "index": i}
                                                for i in range(BASE)]) for r in out["rows"]]
        deployed = dict.fromkeys(SYSTEMS, base_pred)
        experience = {s: [] for s in SYSTEMS}
        for p, phase in enumerate(PHASES):
            # 1. live through the phase with the currently deployed predictor (experience)
            jobs = [{"kind": "experience", "system": s, "phase": phase, "partition": "T", "index": 1000 * (p + 1) + i,
                     "predictor": deployed[s]} for s in SYSTEMS for i in range(EXPERIENCE)]
            res = pool.map(run_job, jobs, chunksize=1)
            new = {s: [row for r in res if r["system"] == s for row in r["rows"]] for s in SYSTEMS}
            # 2. updates
            for s in ("L1", "L2", "L2c", "L3"):
                fresh = corrupt(new[s], rng) if s == "L2c" else new[s]
                experience[s] += fresh
                data = fresh if s == "L3" else base + experience[s]
                cand = train(pool, data, rng)
                decision = "deployed"
                if s in ("L2", "L2c"):
                    vj = [{"kind": "eval", "system": tag, "phase": phase, "partition": "V", "index": 100 * p + i,
                           "predictor": pr} for tag, pr in (("current", deployed[s]), ("candidate", cand))
                          for i in range(VALIDATION)]
                    vr = pool.map(run_job, vj, chunksize=1)
                    tot = {tag: {k: sum(r["totals"][k] for r in vr if r["system"] == tag)
                                 for k in ("J", "collisions", "minutes")} for tag in ("current", "candidate")}
                    ok_j = tot["candidate"]["J"] >= tot["current"]["J"]
                    ok_c = (tot["candidate"]["collisions"] / tot["candidate"]["minutes"]
                            <= 1.05 * tot["current"]["collisions"] / tot["current"]["minutes"])
                    decision = "promoted" if (ok_j and ok_c) else "rejected"
                    log.append({"phase": phase, "system": s, "validation": tot, "decision": decision})
                if decision != "rejected":
                    deployed[s] = cand
                log.append({"phase": phase, "system": s, "event": decision, "train_rows": len(data)})
            # 3. frozen test suites: current phase and the first phase
            suites = [phase] if phase == "PH1" else [phase, "PH1"]
            ej = [{"kind": "eval", "system": s, "phase": su, "suite": su, "partition": args.partition,
                   "index": 10_000 * (list(PHASES).index(su) + 1) + i, "predictor": deployed[s]}
                  for s in SYSTEMS for su in suites for i in range(args.n)]
            for r in pool.map(run_job, ej, chunksize=2):
                evals.append({**r, "after_phase": phase})
            print(phase, "done", round(time.time() - start), flush=True)
    with open(args.out / f"evals_{args.partition}.jsonl", "w", encoding="utf-8") as fh:
        for r in evals:
            fh.write(json.dumps(r) + "\n")
    write_json(args.out / f"run_meta_{args.partition}.json",
               {"provenance": provenance, "log": log, "n": args.n, "phases": PHASES,
                "wall_s": round(time.time() - start, 1)})
    print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
