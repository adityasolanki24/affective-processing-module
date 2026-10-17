"""G6 analysis (prereg/G6_ADDENDUM.md): per condition, every system vs A0 (non-inferiority +10% / −2 pp and the
collision ratio), absolute limits, and the eligibility rule: non-inferior to A0 in ≥ 80% of held-out conditions and
never more than 2× A0's collision rate. Neural systems use the two-level bootstrap over their training seeds.

    python scripts/g6_analyze.py --partition X
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from apm_select.provenance import write_json  # noqa: E402
from apm_select.stats import compare_systems, non_inferior, strip  # noqa: E402


def load(path: Path) -> dict:
    rows: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[r["condition"]][r["label"]][r["seed"]][r["index"]] = r["result"]
    return rows


def arrays(by_seed: dict, idx: list[int]) -> dict:
    seeds = sorted(by_seed)
    keys = by_seed[seeds[0]][idx[0]].keys()
    return {k: np.array([[by_seed[s][i][k] for i in idx] for s in seeds]) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/G6"))
    ap.add_argument("--ref", default="A0")
    args = ap.parse_args()
    data = load(args.runs / args.partition / "lifetimes.jsonl")
    res: dict = {"partition": args.partition, "conditions": {}, "eligibility": {}}
    systems = sorted({s for d in data.values() for s in d})
    for cond, d in data.items():
        row = {}
        for s in systems:
            idx = sorted(set.intersection(*(set(v) for v in d[s].values())) & set(d[args.ref][0]))
            a = arrays(d[s], idx)
            tot = {k: float(v.sum()) for k, v in a.items()}
            row[s] = {"collision_rate": tot["collisions"] / tot["minutes"],
                      "completion": tot["goals_completed"] / tot["goals_issued"],
                      "severe_rate": tot["severe"] / tot["minutes"], "idle_fraction": tot["idle"] / tot["periods"]}
            if s != args.ref:
                c = compare_systems(a, arrays(d[args.ref], idx))
                ni = non_inferior(c, 0.10, 0.02)
                row[s]["vs_ref"] = {"collision_rel": c["collision_rate"]["rel"],
                                    "completion_abs": c["completion"]["abs"], "noninferior": ni["non_inferior"]}
        for s in systems:
            if s != args.ref and "vs_ref" in row[s]:
                ratio = row[s]["collision_rate"] / max(1e-9, row[args.ref]["collision_rate"])
                row[s]["vs_ref"]["collision_ratio"] = ratio
        res["conditions"][cond] = row
    held_out = [c for c in data if c != "nominal"]
    for s in systems:
        if s == args.ref:
            continue
        ni = [res["conditions"][c][s]["vs_ref"]["noninferior"] for c in held_out]
        worst = max(res["conditions"][c][s]["vs_ref"]["collision_ratio"] for c in held_out)
        res["eligibility"][s] = {"noninferior_share": float(np.mean(ni)), "worst_collision_ratio": worst,
                                 "eligible": bool(np.mean(ni) >= 0.8 and worst <= 2.0),
                                 "failed_conditions": [c for c, ok in zip(held_out, ni, strict=True) if not ok]}
    print(write_json(HERE / "results" / "G6" / f"g6_{args.partition}.json", strip(res)))
    print(json.dumps(res["eligibility"], indent=1))


if __name__ == "__main__":
    main()
