"""G3 closed-loop analysis (prereg/G3_ADDENDUM.md). Primary family (X), Holm over four one-sided p-values:
  Q1  best_state (validation choice among the stateful kinds) beats `none`      E2 rule: p_A, p_B
  Q2  `fastslow` beats best_single (validation choice among ema/lds/gru)        E2 rule: p_A, p_B
Secondary: every kind vs none, controls (reset, shuffled) vs live, A0_R2 (F's predictor) vs none, compute.

    python scripts/g3_analyze.py --partition P --power
    python scripts/g3_analyze.py --partition X
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from g1_analyze import describe, load, pair  # noqa: E402

from apm_select.provenance import write_json  # noqa: E402
from apm_select.stats import e2_rule, holm, strip  # noqa: E402

Z = 1.96 + 0.8416


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/G3"))
    ap.add_argument("--power", action="store_true")
    args = ap.parse_args()
    meta = json.loads((args.runs / args.partition / "run_meta.json").read_text())
    d = load(args.runs / args.partition / "lifetimes.jsonl")["nominal"]
    best, single = meta["selection"]["best_state"], meta["selection"]["best_single"]
    if args.power:
        out = {}
        for a, b in ((best, "none"), ("fastslow", single)):
            c = pair(d, a, b)
            n, sc, sp = c["lifetimes"], c["collision_rate"]["rel"]["se"], c["completion"]["abs"]["se"]
            out[f"{a}_vs_{b}"] = {"lifetimes": n, "se_collision_rel": sc, "se_completion_abs": sp,
                                  "required": int(np.ceil(max(n * (Z * sc / 0.10) ** 2, n * (Z * sp / 0.03) ** 2)))}
        print(json.dumps(out, indent=1))
        return
    q1, q2 = pair(d, best, "none"), pair(d, "fastslow", single)
    pv = {"Q1A": q1["collision_rate"]["rel"]["p_not_lower"], "Q1B": q1["completion"]["abs"]["p_not_higher"],
          "Q2A": q2["collision_rate"]["rel"]["p_not_lower"], "Q2B": q2["completion"]["abs"]["p_not_higher"]}
    ph = holm(pv)
    res = {"partition": args.partition, "selection": meta["selection"],
           "descriptives": {s: describe(rows) for s, rows in d.items()},
           "primary": {"p": pv, "p_holm": ph,
                       "Q1_best_state_vs_none": {"comparison": q1, "decision": e2_rule(q1, ph["Q1A"], ph["Q1B"])},
                       "Q2_fastslow_vs_best_single": {"comparison": q2,
                                                      "decision": e2_rule(q2, ph["Q2A"], ph["Q2B"])}},
           "secondary": {}}
    for s in sorted(d):
        if s != "none":
            res["secondary"][f"{s}_vs_none"] = pair(d, s, "none")
        base = s.rsplit("_", 1)[0]
        if s.endswith(("_reset", "_shuffled")) and base in d:
            res["secondary"][f"{s}_vs_{base}"] = pair(d, s, base)
    print(write_json(HERE / "results" / "G3" / f"g3_{args.partition}.json", strip(res)))
    print(json.dumps({"p_holm": ph, "Q1": res["primary"]["Q1_best_state_vs_none"]["decision"],
                      "Q2": res["primary"]["Q2_fastslow_vs_best_single"]["decision"]}, indent=1))


if __name__ == "__main__":
    main()
