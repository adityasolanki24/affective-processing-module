"""E2 tuning (MASTER_PREREGISTRATION.md §1.2): B1 λ grid; B2 the same λ grid × k_pad ∈ {0.5, 1, 2} × k_t ∈ {0.1,
0.25, 0.5}. Tuning lifetimes: E2 partition V, indices 100–129 (disjoint from the predictor's early-stopping lifetimes
0–39), 4 episodes each. Selection: highest mean J. Writes configs/E2_tuned.json.

    python scripts/e2_tune.py --predictor runs/E2/predictor.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from e2_common import run_jobs  # noqa: E402

from apm_research.provenance import write_json  # noqa: E402

LAM_C = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LAM_S = (0.0, 0.5, 2.0)
K_PAD = (0.5, 1.0, 2.0)
K_T = (0.1, 0.25, 0.5)
TUNE = range(100, 130)
EPISODES = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="runs/E2/predictor.json")
    ap.add_argument("--experiment", default="E2")
    ap.add_argument("--out", type=Path, default=HERE / "configs" / "E2_tuned.json")
    args = ap.parse_args()
    start = time.time()
    specs = [{"system": "B1", "lam_c": c, "lam_s": s} for c, s in itertools.product(LAM_C, LAM_S)]
    specs += [{"system": "B2", "lam_c": c, "lam_s": s, "k_pad": kp, "k_t": kt}
              for c, s, kp, kt in itertools.product(LAM_C, LAM_S, K_PAD, K_T)]
    for i, s in enumerate(specs):
        s["label"] = f"cfg{i}"
    jobs = [{"spec": s, "experiment": args.experiment, "partition": "V", "index": i, "n_episodes": EPISODES}
            for s in specs for i in TUNE]
    jobs.sort(key=lambda j: j["spec"]["system"] != "B2")  # long jobs first
    results = run_jobs(jobs, args.predictor)
    table = []
    for s in specs:
        rows = [r["result"] for r in results if r["label"] == s["label"]]
        table.append({**s, "J": float(np.mean([r["J"] for r in rows])),
                      "completion": float(np.mean([r["completion"] for r in rows])),
                      "collision_rate": float(np.mean([r["collision_rate"] for r in rows])),
                      "idle_fraction": float(np.mean([r["idle_fraction"] for r in rows]))})
    best = {sys_: max((t for t in table if t["system"] == sys_), key=lambda t: t["J"]) for sys_ in ("B1", "B2")}
    write_json(args.out, {"best": best, "grid": table, "tuning_lifetimes": [TUNE.start, TUNE.stop],
                          "episodes": EPISODES, "wall_s": round(time.time() - start, 1)})
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
