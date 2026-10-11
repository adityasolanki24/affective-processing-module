"""E4 analysis (MASTER_PREREGISTRATION.md §2 E4): B3 must beat both B4 and B1 under the E2 rule (A or B, with
anti-freezing), two-level bootstrap over test lifetimes and training seeds, Holm over the four one-sided p-values of
the two comparisons (stricter than Holm over two comparisons; see prereg/E4_ADDENDUM.md).

    python scripts/e4_analyze.py --partition X
    python scripts/e4_analyze.py --partition P --power      # pilot: SEs and required test lifetimes only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from e2_analyze import KEYS  # noqa: E402

from apm_research.provenance import write_json  # noqa: E402
from apm_research.stats import holm, paired_lifetimes_two_level  # noqa: E402

PRIMARY = (("B3", "B4"), ("B3", "B1"))


def load(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows: dict[str, dict[int, dict[int, dict]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.setdefault(r["label"], {}).setdefault(r["seed"], {})[r["index"]] = r["result"]
    out = {}
    for label, by_seed in rows.items():
        seeds = sorted(by_seed)
        idx = sorted(set.intersection(*(set(v) for v in by_seed.values())))
        out[label] = {k: np.array([[by_seed[s][i][k] for i in idx] for s in seeds], float) for k in KEYS}
        out[label]["index"], out[label]["seeds"] = np.array(idx), np.array(seeds)
        out[label]["actions"] = np.sum([by_seed[s][i]["actions"] for s in seeds for i in idx], axis=0)
    return out


def describe(d: dict[str, np.ndarray]) -> dict:
    m = d["minutes"].sum()
    per_seed_J = d["J"].mean(1)
    return {"seeds": int(len(d["seeds"])), "lifetimes": int(d["minutes"].shape[1]),
            "completion": float(d["goals_completed"].sum() / d["goals_issued"].sum()),
            "collisions_per_min": float(d["collisions"].sum() / m), "severe_per_min": float(d["severe"].sum() / m),
            "careless_collisions_per_min": float(d["careless_collisions"].sum() / m),
            "interventions_per_min": float(d["interventions"].sum() / m),
            "idle_fraction": float(d["idle"].sum() / d["periods"].sum()), "J_mean": float(d["J"].mean()),
            "J_mean_per_seed": per_seed_J.round(3).tolist(),
            "action_fractions": (d["actions"] / d["actions"].sum()).round(4).tolist()}


def compare(data, a: str, b: str) -> dict:
    keep = np.intersect1d(data[a]["index"], data[b]["index"])
    sel = {s: {k: v[:, np.isin(data[s]["index"], keep)] for k, v in data[s].items() if k in KEYS} for s in (a, b)}
    return paired_lifetimes_two_level(sel[a], sel[b])


def rule(c: dict, p_a: float, p_b: float) -> dict:
    col, comp, idle = c["collision_rate_relative"], c["completion_difference"], c["idle_difference"]
    a = col["estimate"] <= -0.10 and p_a < 0.025 and comp["ci95"][0] >= -0.03
    b = comp["estimate"] >= 0.05 and p_b < 0.025 and col["ci95"][1] <= 0.05
    no_freeze = idle["estimate"] <= 0.05
    return {"criterion_A": bool(a), "criterion_B": bool(b), "no_degenerate_freezing": bool(no_freeze),
            "beats": bool((a or b) and no_freeze)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/E4"))
    ap.add_argument("--power", action="store_true")
    args = ap.parse_args()
    data = load(args.runs / f"eval_{args.partition}" / "lifetimes.jsonl")
    primary = {f"{a}_vs_{b}": compare(data, a, b) for a, b in PRIMARY if a in data and b in data}
    if args.power:
        need = {}
        for k, c in primary.items():
            n = len(data["B1"]["index"])
            se_col, se_comp = c["collision_rate_relative"]["se"], c["completion_difference"]["se"]
            need[k] = {"lifetimes": n, "se_collision_relative": se_col, "se_completion_pp": se_comp,
                       "required_lifetimes_80pct_power": int(np.ceil(max(n * (2.8 * se_col / 0.10) ** 2,
                                                                         n * (2.8 * se_comp / 0.03) ** 2)))}
        print(json.dumps(need, indent=1))
        return
    p = holm({f"{k}:{t}": c[f"p_{q}"] for k, c in primary.items()
              for t, q in (("A", "collision_not_lower"), ("B", "completion_not_higher"))})
    for k, c in primary.items():
        c["decision"] = rule(c, p[f"{k}:A"], p[f"{k}:B"]) | {"p_holm": {"A": p[f"{k}:A"], "B": p[f"{k}:B"]}}
    credible = all(primary[f"{a}_vs_{b}"]["decision"]["beats"] for a, b in PRIMARY)
    verdict = ("credible candidate" if credible else
               "no evidence for a specialised regulation structure (B3 beats B1 but not B4)"
               if primary["B3_vs_B1"]["decision"]["beats"] else "not a credible candidate")
    result = {"partition": args.partition, "systems": {s: describe(d) for s, d in data.items()},
              "primary": primary, "B3_credible_candidate": bool(credible), "verdict": verdict, "secondary": {}}
    pairs = [("B3", "B2"), ("B4", "B1"), ("B2", "B1")] + [(s, "B3") for s in data if s.startswith("B3_")] + \
            [(s, "B4") for s in data if s.startswith("B4_")]
    for a, b in pairs:
        if a in data and b in data:
            c = compare(data, a, b)
            c["decision_unadjusted"] = rule(c, c["p_collision_not_lower"], c["p_completion_not_higher"])
            result["secondary"][f"{a}_vs_{b}"] = c
    print(write_json(HERE / "results" / "E4" / f"E4_{args.partition}.json", result))
    print(json.dumps({"systems": result["systems"], "verdict": verdict,
                      "primary": {k: {m: c[m] for m in ("collision_rate_relative", "completion_difference",
                                                         "idle_difference", "decision")}
                                  for k, c in primary.items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
