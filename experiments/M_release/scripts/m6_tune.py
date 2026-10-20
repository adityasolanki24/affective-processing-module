"""M6 calibration (PROTOCOL.md §3): pick MONOTONE* per robot from its partition-V lifetimes.

    python experiments/M_release/scripts/m6_tune.py --robot nova_carter --runs <V dirs>

The rule (frozen in §3): the lowest mean realised cost (§6), averaged over the robot's V lifetimes; ties go to the
earlier configuration in the protocol's order. Writes or updates `configs/M6_tuned.json`, which must be committed
before any X run that uses MONOTONE*.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m6_analyze import enc_cost, load  # noqa: E402

CFG = HERE.parents[0] / "configs"
ORDER = ["MONOTONE:lo=0.1/width=0.2", "MONOTONE:lo=0.1/width=0.4", "MONOTONE:lo=0.2/width=0.2",
         "MONOTONE:lo=0.2/width=0.4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="handoff")
    ap.add_argument("--experiment", default="M6", help="seed space of the V lifetimes (R: remediation re-tune)")
    ap.add_argument("--config", default="M6_tuned.json", help="output; M6_tuned_R.json for the remediation")
    a = ap.parse_args()
    task = json.loads((CFG / f"M5_{a.robot}_tasks.json").read_text())["tasks"][a.task]
    recs = [r for r in load(a.runs) if r["robot"] == a.robot and r["partition"] == "V"
            and r.get("experiment", "M6") == a.experiment]
    table = {}
    for cfg in ORDER:
        lts = [np.mean([enc_cost(x, task) for x in r["rows"]]) for r in recs if r["system"] == cfg and r["rows"]]
        table[cfg] = {"lifetimes": len(lts), "mean_cost": round(float(np.mean(lts)), 3) if lts else None}
    valid = [c for c in ORDER if table[c]["mean_cost"] is not None]
    best = min(valid, key=lambda c: (table[c]["mean_cost"], ORDER.index(c)))
    path = CFG / a.config
    tuned = json.loads(path.read_text()) if path.exists() else {}
    tuned[a.robot] = best
    tuned.setdefault("_tables", {})[a.robot] = table
    tuned["_rule"] = ("PROTOCOL.md §3: lowest mean realised cost on partition V (warehouse, change_B and late_B)"
                      + ("; remediation: experiment R, matched supervision" if a.experiment == "R" else ""))
    path.write_text(json.dumps(tuned, indent=1))
    print(a.robot, "MONOTONE* =", best)
    for c in ORDER:
        print(f"  {c:28} {table[c]}")


if __name__ == "__main__":
    main()
