"""G1 equal-budget tuning on partition V (prereg/G1_ADDENDUM.md fixes K, N, episodes and the sampling seed).

Every family R1–R4, R6–R8 gets the same K configurations-worth of budget: K Latin-hypercube draws from its
preregistered space (`apm_select.tuning`), each evaluated on the same N tuning lifetimes (G1-V 0…N−1) with common
random numbers. Selection: the configuration with the highest mean tuning utility J (programme F §4.3). Every
configuration's metrics are kept (validation Pareto front). R5 is trained separately (g1_es.py) with the same number
of lifetime evaluations.

    python scripts/g1_tune.py --k 48 --n 30 --episodes 10 --workers 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from g_common import run_jobs  # noqa: E402

from apm_research.provenance import environment, write_json  # noqa: E402
from apm_select.tuning import FAMILIES, family_configs  # noqa: E402

SAMPLE_SEED = 20261012
SUMMARY = ("J", "collisions", "severe", "goals_completed", "goals_issued", "minutes", "idle", "periods", "distance",
           "person_collisions", "person_encounters", "hesitant", "decisions", "switches")


def summarise(rows: list[dict]) -> dict:
    t = {k: float(np.sum([r["totals"][k] for r in rows])) for k in SUMMARY}
    return {"lifetimes": len(rows), "J_mean": float(np.mean([r["totals"]["J"] for r in rows])),
            "J_se": float(np.std([r["totals"]["J"] for r in rows], ddof=1) / np.sqrt(len(rows))),
            "collision_rate": t["collisions"] / t["minutes"], "severe_rate": t["severe"] / t["minutes"],
            "completion": t["goals_completed"] / t["goals_issued"], "idle_fraction": t["idle"] / t["periods"],
            "collisions_per_encounter": t["person_collisions"] / max(1.0, t["person_encounters"]),
            "hesitation": t["hesitant"] / t["decisions"], "switch_rate": t["switches"] / t["decisions"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--families", default="R1,R2,R3,R4,R6,R7,R8")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G1/tune"))
    args = ap.parse_args()
    start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    specs = [s for f in args.families.split(",") for s in family_configs(f, args.k, SAMPLE_SEED)]
    jobs = [{"spec": s, "experiment": "G1", "partition": "V", "index": i, "n_episodes": args.episodes}
            for s in specs for i in range(args.n)]
    jobs.sort(key=lambda j: j["spec"]["kind"] != "b2v")  # slow jobs first
    results = run_jobs(jobs, args.workers, args.out / "lifetimes.jsonl")
    by = defaultdict(list)
    for r in results:
        by[r["config_id"]].append(r)
    table = [{**s, **summarise(by[s["config_id"]])} for s in specs]
    best = {}
    for f in args.families.split(","):
        rows = [t for t in table if t["family"] == f]
        best[f] = max(rows, key=lambda t: t["J_mean"])
    write_json(HERE / "results" / "G1" / "tuning_table.json",
               {"k": args.k, "n": args.n, "episodes": args.episodes, "sample_seed": SAMPLE_SEED, "table": table,
                "environment": environment(), "wall_s": round(time.time() - start, 1)})
    write_json(HERE / "configs" / "G1_selected_partial.json",
               {"selected": best, "rule": "max mean J on G1-V lifetimes 0..n-1", "k": args.k, "n": args.n})
    print(json.dumps({f: {k: b[k] for k in ("config_id", "J_mean", "collision_rate", "completion")}
                      for f, b in best.items()}, indent=1))
    assert set(FAMILIES) >= set(best)


if __name__ == "__main__":
    main()
