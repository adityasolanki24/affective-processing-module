"""G8 equal-budget λ retuning of every A0 variant (prereg/G8_ADDENDUM.md): K Latin-hypercube (λc, λs) per variant on
G8-V lifetimes 0…N−1 × 10 episodes, selection by mean J; everything else of the variant fixed.

    python scripts/g8_tune.py --variants configs/G8_variants.json --k 48 --n 30 --workers 90
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
from g1_tune import summarise  # noqa: E402
from g_common import run_jobs  # noqa: E402

from apm_research.provenance import environment, write_json  # noqa: E402
from apm_select.tuning import LAM, lhs  # noqa: E402

SEED = 20261014


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G8/tune"))
    args = ap.parse_args()
    start = time.time()
    variants = json.loads(args.variants.read_text())["variants"]
    grid = lhs(LAM, args.k, SEED)
    specs = [{**v, "lam_c": g["lam_c"], "lam_s": g["lam_s"], "label": name, "config_id": f"{name}:{i}"}
             for name, v in variants.items() for i, g in enumerate(grid)]
    jobs = [{"spec": s, "experiment": "G8", "partition": "V", "index": i, "n_episodes": 10}
            for s in specs for i in range(args.n)]
    jobs.sort(key=lambda j: j["spec"].get("memory") is None)
    args.out.mkdir(parents=True, exist_ok=True)
    res = run_jobs(jobs, args.workers, args.out / "lifetimes.jsonl")
    by = defaultdict(list)
    for r in res:
        by[r["config_id"]].append(r)
    table = [{**s, **summarise(by[s["config_id"]])} for s in specs]
    best = {name: max((t for t in table if t["label"] == name), key=lambda t: t["J_mean"]) for name in variants}
    write_json(HERE / "results" / "G8" / "tuning_table.json", {"table": table, "environment": environment(),
                                                                "wall_s": round(time.time() - start, 1)})
    keep = {k for v in variants.values() for k in v} | {"lam_c", "lam_s"}
    systems = {name: {k: b[k] for k in keep if k in b} for name, b in best.items()}
    out = {"rule": "per variant max mean J on G8-V (equal K)", "systems": systems,
           "validation": {n: {k: best[n][k] for k in ("config_id", "J_mean", "collision_rate", "completion")}
                          for n in best}}
    (HERE / "configs" / "G8_selected.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["validation"], indent=1))
    assert np.isfinite([b["J_mean"] for b in best.values()]).all()


if __name__ == "__main__":
    main()
