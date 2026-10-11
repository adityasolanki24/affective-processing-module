"""G9 validity re-test analysis (prereg/G9_ADDENDUM.md): the preregistered primary families of G1, G3, G4/G5, G8 and the
G6 eligibility rule, recomputed on fresh lifetimes with **avoidable** collisions (contacts after the first step of an
episode; testbed defect D8). Implementation: every per-lifetime `collisions` total is replaced by
`avoidable_collisions` before the unchanged comparison code runs; all other metrics are untouched. The total-collision
results on the same lifetimes are reported alongside.

    python scripts/g9_analyze.py --runs runs/G9 [--metric avoidable|total]
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
from apm_select.stats import compare_systems, e2_rule, holm, non_inferior, strip  # noqa: E402


def swap(t: dict, metric: str) -> dict:
    t = dict(t)
    if metric == "avoidable":
        t["collisions"] = t["avoidable_collisions"]
    return t


def load_rule(path: Path, metric: str) -> dict:
    """g1_run / g3_closed output → {label: {index: totals}} (nominal condition)."""
    d: dict = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("condition", "nominal") == "nominal":
            d[r["label"]][r["index"]] = swap(r["totals"], metric)
    return d


def arr(rows: dict, idx: list[int]) -> dict:
    keys = rows[idx[0]].keys()
    return {k: np.array([rows[i][k] for i in idx], float) for k in keys}


def pair(d: dict, a: str, b: str) -> dict:
    idx = sorted(set(d[a]) & set(d[b]))
    return compare_systems(arr(d[a], idx), arr(d[b], idx))


def brief(c: dict) -> dict:
    return {k: {"estimate": c[m][x]["estimate"], "ci95": c[m][x]["ci95"]}
            for k, m, x in (("collision_rel", "collision_rate", "rel"), ("completion_abs", "completion", "abs"),
                            ("idle_abs", "idle_fraction", "abs"))}


def superiority(c: dict) -> tuple[float, float]:
    return c["collision_rate"]["rel"]["p_not_lower"], c["completion"]["abs"]["p_not_higher"]


def g1(d: dict) -> dict:
    p1, p2, p3 = pair(d, "R6", "R1"), pair(d, "R2", "R6"), pair(d, "R2", "R1")
    ni = non_inferior(p2, 0.10, 0.02)
    pv = {"P1A": superiority(p1)[0], "P1B": superiority(p1)[1], "P2": ni["p"], "P3A": superiority(p3)[0],
          "P3B": superiority(p3)[1]}
    ph = holm(pv)
    return {"p_holm": ph, "P1_R6_beats_R1": {**brief(p1), "decision": e2_rule(p1, ph["P1A"], ph["P1B"])},
            "P2_R2_noninferior_R6": {**brief(p2), "holds": bool(ni["non_inferior"] and ph["P2"] < 0.05)},
            "P3_R2_beats_R1": {**brief(p3), "decision": e2_rule(p3, ph["P3A"], ph["P3B"])},
            "secondary": {f"{a}_vs_{b}": brief(pair(d, a, b)) for a, b in
                          (("R1", "R0"), ("R7", "R6"), ("R8", "R6"), ("R3", "R1"), ("R4", "R1"), ("R5", "R1"),
                           ("R7", "R1"), ("R8", "R1")) if a in d and b in d}}


def g3(d: dict, sel: dict) -> dict:
    best, single = sel["best_state"], sel["best_single"]
    q1, q2 = pair(d, best, "none"), pair(d, "fastslow", single)
    pv = {"Q1A": superiority(q1)[0], "Q1B": superiority(q1)[1], "Q2A": superiority(q2)[0], "Q2B": superiority(q2)[1]}
    ph = holm(pv)
    return {"p_holm": ph, "Q1": {**brief(q1), "decision": e2_rule(q1, ph["Q1A"], ph["Q1B"])},
            "Q2": {**brief(q2), "decision": e2_rule(q2, ph["Q2A"], ph["Q2B"])},
            "secondary": {f"{s}_vs_none": brief(pair(d, s, "none")) for s in d if s != "none"}}


def g8(d: dict) -> dict:
    out, pv = {}, {}
    for v in ("A0_static", "A0_nomem", "A0_single", "A0_heur"):
        c = pair(d, v, "A0")
        ni = non_inferior(c, 0.10, 0.02)
        out[v] = {**brief(c), "ni": ni}
        pv[v] = ni["p"]
    c = pair(d, "A0_mem", "A0")
    pa, pb = superiority(c)
    out["A0_mem"] = brief(c)
    pv["A0_mem"] = min(pa, pb)
    ph = holm(pv)
    for v in ("A0_static", "A0_nomem", "A0_single", "A0_heur"):
        out[v]["component_unjustified"] = bool(out[v]["ni"]["non_inferior"] and ph[v] < 0.05)
    out["A0_mem"]["decision"] = e2_rule(c, ph["A0_mem"] if pa <= pb else 1.0, ph["A0_mem"] if pb < pa else 1.0)
    return {"p_holm": ph, "tests": out}


def load_seeded(path: Path, metric: str, key: str = "result") -> dict:
    """g45_eval / g6_run output → {condition: {label: {seed: {index: totals}}}}."""
    d: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        d[r.get("condition", "nominal")][r["label"]][r["seed"]][r["index"]] = swap(r[key], metric)
    return d


def arr2(by_seed: dict, idx: list[int]) -> dict:
    seeds = sorted(by_seed)
    keys = by_seed[seeds[0]][idx[0]].keys()
    return {k: np.array([[by_seed[s][i][k] for i in idx] for s in seeds], float) for k in keys}


def pair2(d: dict, a: str, b: str) -> dict:
    idx = sorted(set.intersection(*(set(v) for v in d[a].values())) & set(d[b][0]))
    return compare_systems(arr2(d[a], idx), arr2(d[b], idx))


def g45(d: dict) -> dict:
    cells = ("A2_bcppo", "A3_bcppo", "A4_ppo")
    comps = {c: pair2(d, c, "A0") for c in cells}
    pv = {}
    for c, x in comps.items():
        pv[f"{c}:A"], pv[f"{c}:B"] = superiority(x)
    ph = holm(pv)
    g4 = {c: {**brief(x), "decision": e2_rule(x, ph[f"{c}:A"], ph[f"{c}:B"])} for c, x in comps.items()}
    m = pair2(d, "A3_bcppo", "A3_ppo")
    gru = comps["A3_bcppo"]
    ni = non_inferior(gru, 0.10, 0.02)
    p5 = holm({"A": superiority(m)[0], "B": superiority(m)[1], "NI": ni["p"]})
    return {"G4_p_holm": ph, "G4": g4,
            "G5_bcppo_vs_ppo": {**brief(m), "decision": e2_rule(m, p5["A"], p5["B"])},
            "G5_best_gru_noninferior_A0": {**brief(gru), "holds": bool(ni["non_inferior"] and p5["NI"] < 0.05)},
            "secondary": {"A1_vs_A0": brief(pair2(d, "A1", "A0")), "A4_vs_A2": brief(pair2(d, "A4_ppo", "A2_bcppo"))}}


def g6(data: dict) -> dict:
    systems = sorted({s for c in data.values() for s in c} - {"A0"})
    held = [c for c in data if c != "nominal"]
    res = {}
    for s in systems:
        ni, worst, per = [], 0.0, {}
        for c in held:
            x = pair2(data[c], s, "A0")
            ok = non_inferior(x, 0.10, 0.02)["non_inferior"]
            ni.append(ok)
            worst = max(worst, x["collision_rate"]["a"] / max(1e-9, x["collision_rate"]["b"]))
            per[c] = {**brief(x), "noninferior": ok}
        res[s] = {"noninferior_share": float(np.mean(ni)), "worst_collision_ratio": worst,
                  "eligible": bool(np.mean(ni) >= 0.8 and worst <= 2.0), "per_condition": per}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs/G9"))
    ap.add_argument("--metric", default="avoidable", choices=("avoidable", "total"))
    args = ap.parse_args()
    out: dict = {"metric": args.metric}
    r = args.runs
    if (r / "G1" / "X" / "lifetimes.jsonl").exists():
        out["G1"] = g1(load_rule(r / "G1" / "X" / "lifetimes.jsonl", args.metric))
    if (r / "G3" / "X" / "lifetimes.jsonl").exists():
        sel = json.loads((HERE / "configs" / "G3_selected.json").read_text())
        out["G3"] = g3(load_rule(r / "G3" / "X" / "lifetimes.jsonl", args.metric), sel)
    if (r / "G8" / "X" / "lifetimes.jsonl").exists():
        out["G8"] = g8(load_rule(r / "G8" / "X" / "lifetimes.jsonl", args.metric))
    if (r / "G8" / "S" / "lifetimes.jsonl").exists():
        d = load_rule(r / "G8" / "S" / "lifetimes.jsonl", args.metric)
        c = pair(d, "A0_min", "A0")
        out["G8_stage2"] = {**brief(c), "ni": non_inferior(c, 0.10, 0.02)}
    if (r / "G45" / "eval_X" / "lifetimes.jsonl").exists():
        out["G45"] = g45(load_seeded(r / "G45" / "eval_X" / "lifetimes.jsonl", args.metric)["nominal"])
    if (r / "G6" / "X" / "lifetimes.jsonl").exists():
        out["G6"] = g6(load_seeded(r / "G6" / "X" / "lifetimes.jsonl", args.metric))
    print(write_json(HERE / "results" / "G9" / f"g9_{args.metric}.json", strip(out)))


if __name__ == "__main__":
    main()
