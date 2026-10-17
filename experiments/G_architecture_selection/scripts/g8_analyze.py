"""G8 analysis (prereg/G8_ADDENDUM.md). Stage 1 (X): each removal variant non-inferior to A0 (+10% / −2 pp), A0_mem
superior to A0 (E2 rule); Holm over five p-values. Stage 2 (S): A0_min non-inferior to A0. Complexity table.

    python scripts/g8_analyze.py --partition X
    python scripts/g8_analyze.py --partition S
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from g1_analyze import describe, load, pair  # noqa: E402

from apm_select.provenance import write_json  # noqa: E402
from apm_select.stats import e2_rule, holm, non_inferior, strip  # noqa: E402

REMOVALS = ("A0_static", "A0_nomem", "A0_single", "A0_heur")
PARAMS = {"A0": 36495, "A0_static": 36495, "A0_nomem": 36495, "A0_single": 7299, "A0_heur": 0, "A0_mem": 36495,
          "A0_min": None, "B0": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/G8"))
    args = ap.parse_args()
    d = load(args.runs / args.partition / "lifetimes.jsonl")["nominal"]
    res: dict = {"partition": args.partition, "descriptives": {s: describe(r) for s, r in d.items()}}
    for s, x in res["descriptives"].items():
        x["parameters"] = PARAMS.get(s)
    if args.partition == "X":
        comps, pv = {}, {}
        for v in REMOVALS:
            c = pair(d, v, "A0")
            ni = non_inferior(c, 0.10, 0.02)
            comps[v] = {"comparison": c, "noninferiority": ni}
            pv[v] = ni["p"]
        c = pair(d, "A0_mem", "A0")
        pa, pb = c["collision_rate"]["rel"]["p_not_lower"], c["completion"]["abs"]["p_not_higher"]
        comps["A0_mem"] = {"comparison": c}
        pv["A0_mem"] = min(pa, pb)
        ph = holm(pv)
        for v in REMOVALS:
            comps[v]["component_unjustified"] = bool(comps[v]["noninferiority"]["non_inferior"] and ph[v] < 0.05)
        best_p = "A" if pa <= pb else "B"
        comps["A0_mem"]["decision"] = e2_rule(c, ph["A0_mem"] if best_p == "A" else 1.0,
                                              ph["A0_mem"] if best_p == "B" else 1.0)
        res["primary"] = {"p": pv, "p_holm": ph, "tests": comps}
        res["A0_min_recipe"] = {"remove": [v for v in REMOVALS if comps[v]["component_unjustified"]],
                                "add_memory": comps["A0_mem"]["decision"]["beats"]}
    else:
        c = pair(d, "A0_min", "A0")
        res["stage2"] = {"comparison": c, "noninferiority": non_inferior(c, 0.10, 0.02)}
    print(write_json(HERE / "results" / "G8" / f"g8_{args.partition}.json", strip(res)))
    if "primary" in res:
        print(json.dumps({"p_holm": res["primary"]["p_holm"], "recipe": res["A0_min_recipe"]}, indent=1))
    for s, x in sorted(res["descriptives"].items()):
        print(f'{s:10s} col {x["collision_rate"]:.4f} comp {x["completion"]:.4f} idle {x["idle_fraction"]:.4f} '
              f'J {x["J_mean"]:.2f} us {x.get("compute_us_per_decision", 0):.0f} params {x["parameters"]}')


if __name__ == "__main__":
    main()
