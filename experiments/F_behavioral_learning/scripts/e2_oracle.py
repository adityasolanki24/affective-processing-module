"""Oracle reference rates for E2 (evaluation-only system, MASTER_PREREGISTRATION.md §1.2).

The Oracle is B1 whose memory feature for each person is replaced by the true per-encounter collision rate of that
person's hidden disposition. Those rates are estimated here on exploration lifetimes of partition T (the predictor's
training distribution) as the converged memory estimate, averaged over identities of each disposition. Only T data
are used; no evaluation partition is touched.

    python scripts/e2_oracle.py --n 60
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import seeds as S  # noqa: E402

from apm_research.harness import Lifetime, run_lifetime  # noqa: E402
from apm_research.policies import Explore  # noqa: E402
from apm_research.provenance import environment, write_json  # noqa: E402
from apm_testbeds.wheeled_ctrl import child  # noqa: E402


def collect(job):
    experiment, index, episodes = job
    ss = S.world_seed(experiment, "T", index)
    lt = Lifetime(ss, n_episodes=episodes)
    policy = Explore(child(ss, 500))
    run_lifetime(lt, policy)
    rows = []
    for ident in lt.population():
        p, conf = policy.memory.p_collision(ident.name, "person")
        rows.append((ident.disposition, p, conf))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="E2")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--first", type=int, default=200, help="T indices not used by the predictor (0–149)")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    jobs = [(args.experiment, args.first + i, args.episodes) for i in range(args.n)]
    with Pool(4) as pool:
        rows = [r for rs in pool.map(collect, jobs) for r in rs]
    rates = {}
    for d in ("careful", "careless"):
        ps = np.array([p for disp, p, c in rows if disp == d and c > 0])
        rates[d] = {"mean": float(ps.mean()), "sd": float(ps.std()), "identities": int(len(ps))}
    out = {"environment": environment(), "partition": "T", "indices": [args.first, args.first + args.n - 1],
           "episodes": args.episodes, "rates_by_disposition": {d: v["mean"] for d, v in rates.items()},
           "detail": rates}
    print(write_json(HERE / "configs" / "oracle_rates.json", out))
    print(rates)


if __name__ == "__main__":
    main()
