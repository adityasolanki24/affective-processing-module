"""E2 analysis (MASTER_PREREGISTRATION.md E2): paired cluster-bootstrap comparisons over lifetimes.

    python scripts/e2_analyze.py --partition X            # confirmatory
    python scripts/e2_analyze.py --partition P --power    # pilot: SEs and required lifetimes only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from apm_research.provenance import write_json  # noqa: E402
from apm_research.stats import holm, paired_lifetimes  # noqa: E402

KEYS = ("collisions", "minutes", "goals_completed", "goals_issued", "idle", "periods", "severe", "interventions",
        "person_collisions", "careless_collisions", "careful_collisions", "slips", "distance", "goal_time_sum", "J")


def load(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows: dict[str, dict[int, dict]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(r["label"], {})[r["index"]] = r["result"]
    out = {}
    for label, by_index in rows.items():
        idx = sorted(by_index)
        out[label] = {k: np.array([by_index[i][k] for i in idx], dtype=float) for k in KEYS}
        out[label]["index"] = np.array(idx)
        out[label]["actions"] = np.sum([by_index[i]["actions"] for i in idx], axis=0)
        out[label]["fallbacks"] = np.array([by_index[i]["fallbacks"] for i in idx])
    return out


def describe(d: dict[str, np.ndarray]) -> dict:
    m = d["minutes"].sum()
    return {"lifetimes": int(len(d["minutes"])),
            "completion": float(d["goals_completed"].sum() / d["goals_issued"].sum()),
            "collisions_per_min": float(d["collisions"].sum() / m), "severe_per_min": float(d["severe"].sum() / m),
            "careless_collisions_per_min": float(d["careless_collisions"].sum() / m),
            "careful_collisions_per_min": float(d["careful_collisions"].sum() / m),
            "interventions_per_min": float(d["interventions"].sum() / m), "slips_per_min": float(d["slips"].sum() / m),
            "distance_per_min": float(d["distance"].sum() / m),
            "mean_time_to_goal_s": float(d["goal_time_sum"].sum() / max(1, d["goals_completed"].sum())),
            "idle_fraction": float(d["idle"].sum() / d["periods"].sum()), "J_mean": float(d["J"].mean()),
            "action_fractions": (d["actions"] / d["actions"].sum()).round(4).tolist(),
            "fallbacks": int(d["fallbacks"].sum())}


def decide(c: dict, p_holm: dict) -> dict:
    col, comp, idle = c["collision_rate_relative"], c["completion_difference"], c["idle_difference"]
    a = col["estimate"] <= -0.10 and p_holm["A"] < 0.025 and comp["ci95"][0] >= -0.03
    b = comp["estimate"] >= 0.05 and p_holm["B"] < 0.025 and col["ci95"][1] <= 0.05
    no_freeze = idle["estimate"] <= 0.05
    return {"criterion_A": bool(a), "criterion_B": bool(b), "no_degenerate_freezing": bool(no_freeze),
            "beats": bool((a or b) and no_freeze), "p_holm": p_holm}


def compare(data, a: str, b: str, primary: bool = False) -> dict:
    keep = np.intersect1d(data[a]["index"], data[b]["index"])
    sel = {s: {k: v[np.isin(data[s]["index"], keep)] for k, v in data[s].items() if k in KEYS} for s in (a, b)}
    c = paired_lifetimes(sel[a], sel[b])
    p = holm({"A": c["p_collision_not_lower"], "B": c["p_completion_not_higher"]})
    c["decision"] = decide(c, p)
    c["primary"] = primary
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="E2")
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--power", action="store_true")
    args = ap.parse_args()
    data = load(args.runs / args.experiment / args.partition / "lifetimes.jsonl")
    if args.power:
        c = compare(data, "B2", "B1")
        n = len(data["B1"]["minutes"])
        se_col, se_comp = c["collision_rate_relative"]["se"], c["completion_difference"]["se"]
        need = int(np.ceil(max(n * (2.8 * se_col / 0.10) ** 2, n * (2.8 * se_comp / 0.03) ** 2)))
        print(json.dumps({"lifetimes": n, "se_collision_relative": se_col, "se_completion_pp": se_comp,
                          "required_lifetimes_80pct_power": need}, indent=1))
        return
    result = {"partition": args.partition, "systems": {s: describe(d) for s, d in data.items()},
              "primary_B2_vs_B1": compare(data, "B2", "B1", primary=True), "secondary": {}}
    pairs = [("B1", "B0"), ("B2", "B0"), ("Oracle", "B1")] + [(s, "B2") for s in data if s.startswith("B2_")]
    for a, b in pairs:
        if a in data and b in data:
            result["secondary"][f"{a}_vs_{b}"] = compare(data, a, b)
    print(write_json(HERE / "results" / args.experiment / f"{args.experiment}_{args.partition}.json", result))
    print(json.dumps({"systems": result["systems"], "primary": result["primary_B2_vs_B1"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
