"""G1 analysis (MASTER_PREREGISTRATION.md §G1, prereg/G1_ADDENDUM.md).

Primary family (nominal condition, partition X), Holm over five one-sided p-values:
  P1  R6 (fairly tuned B2) beats R1 (fairly tuned static B1)            E2 rule: p_A, p_B
  P2  Rbest (best simple family on validation) is non-inferior to R6     collision r upper 95% < +10% and
                                                                         completion-difference lower 95% > −2 pp: p_NI
  P3  Rbest beats R1                                                     E2 rule: p_A, p_B
Secondary (unadjusted, labelled): everything else, including exposure-normalised rates, hesitation, switching,
compute, negative controls and the shift conditions (partition S).

    python scripts/g1_analyze.py --partition P --power
    python scripts/g1_analyze.py --partition X
    python scripts/g1_analyze.py --partition S
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
from apm_select.stats import METRICS, compare_systems, e2_rule, holm, non_inferior, strip  # noqa: E402

NI_COLLISION, NI_COMPLETION = 0.10, 0.02
Z_NI, Z_SUP, Z_POW = 1.645, 1.96, 0.8416


def load(path: Path) -> dict:
    data: dict = defaultdict(lambda: defaultdict(dict))
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        data[r["condition"]][r["label"]][r["index"]] = r
    return data


def arrays(rows: dict, idx: list[int]) -> dict:
    keys = rows[idx[0]]["totals"].keys()
    return {k: np.array([rows[i]["totals"][k] for i in idx], float) for k in keys}


def pair(cond: dict, a: str, b: str) -> dict:
    idx = sorted(set(cond[a]) & set(cond[b]))
    c = compare_systems(arrays(cond[a], idx), arrays(cond[b], idx))
    c["lifetimes"] = len(idx)
    return c


def describe(rows: dict) -> dict:
    idx = sorted(rows)
    t = {k: float(v.sum()) for k, v in arrays(rows, idx).items()}
    t["collisions_x100"] = 100 * t["collisions"]
    out = {name: t[num] / max(1e-12, t[den]) for name, (num, den) in METRICS.items() if num in t and den in t}
    acts = np.sum([rows[i]["actions"] for i in idx], axis=0)
    out |= {"lifetimes": len(idx), "J_mean": t["J"] / len(idx),
            "action_fractions": (acts / acts.sum()).round(4).tolist()}
    return out


def primary(cond: dict, best: str) -> dict:
    p1, p3 = pair(cond, "R6", "R1"), pair(cond, best, "R1")
    p2 = pair(cond, best, "R6")
    ni = non_inferior(p2, NI_COLLISION, NI_COMPLETION)
    pv = {"P1A": p1["collision_rate"]["rel"]["p_not_lower"], "P1B": p1["completion"]["abs"]["p_not_higher"],
          "P2": ni["p"], "P3A": p3["collision_rate"]["rel"]["p_not_lower"],
          "P3B": p3["completion"]["abs"]["p_not_higher"]}
    ph = holm(pv)
    d1 = e2_rule(p1, ph["P1A"], ph["P1B"])
    d3 = e2_rule(p3, ph["P3A"], ph["P3B"])
    d2 = {**ni, "holds_after_holm": bool(ni["non_inferior"] and ph["P2"] < 0.05)}
    return {"best_simple": best, "p": pv, "p_holm": ph,
            "P1_R6_vs_R1": {"comparison": p1, "decision": d1},
            "P2_best_vs_R6_noninferiority": {"comparison": p2, "decision": d2},
            "P3_best_vs_R1": {"comparison": p3, "decision": d3}}


def power(cond: dict, best: str) -> dict:
    out = {}
    for a, b, kind in (("R6", "R0", "sup"), (best, "R0", "sup"), (best, "R6", "ni")):
        c = pair(cond, a, b)
        n, se_col, se_comp = c["lifetimes"], c["collision_rate"]["rel"]["se"], c["completion"]["abs"]["se"]
        if kind == "sup":
            need = max(n * ((Z_SUP + Z_POW) * se_col / 0.10) ** 2, n * ((Z_SUP + Z_POW) * se_comp / 0.03) ** 2)
        else:
            need = max(n * ((Z_NI + Z_POW) * se_col / NI_COLLISION) ** 2,
                       n * ((Z_NI + Z_POW) * se_comp / NI_COMPLETION) ** 2)
        out[f"{a}_vs_{b}"] = {"lifetimes": n, "se_collision_rel": se_col, "se_completion_abs": se_comp,
                              "rule": kind, "required": int(np.ceil(need))}
    return out


def figure(desc: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, d in desc.items():
        if name == "B0":
            continue
        ctrl = "_" in name
        ax.scatter(d["collision_rate"], d["completion"], marker="x" if ctrl else "o", s=36,
                   color="0.55" if ctrl else "C0" if name.startswith(("R0", "R1")) else "C3" if name in
                   ("R6", "R7", "R8") else "C2")
        ax.annotate(name, (d["collision_rate"], d["completion"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("collisions per minute (lower is better)")
    ax.set_ylabel("goal completion (higher is better)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/G1"))
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--best", default=None, help="override best simple family (pilot only)")
    args = ap.parse_args()
    meta = json.loads((args.runs / args.partition / "run_meta.json").read_text())
    data = load(args.runs / args.partition / "lifetimes.jsonl")
    sel = json.loads(Path(meta["systems_file"]).read_text()) if Path(meta["systems_file"]).exists() else {}
    best = args.best or sel.get("best_simple", "R2")
    if args.power:
        print(json.dumps(power(data["nominal"], best), indent=1))
        return
    res: dict = {"partition": args.partition, "best_simple": best,
                 "descriptives": {c: {s: describe(rows) for s, rows in d.items()} for c, d in data.items()}}
    if "nominal" in data and args.partition == "X":
        res["primary"] = primary(data["nominal"], best)
    sec: dict = {}
    for cond, d in data.items():
        names = sorted(d)
        refs = [r for r in ("R1", "R6") if r in d]
        rows = {}
        for a in names:
            for b in refs:
                if a != b:
                    c = pair(d, a, b)
                    rows[f"{a}_vs_{b}"] = {"comparison": c, "e2_rule_unadjusted": e2_rule(
                        c, c["collision_rate"]["rel"]["p_not_lower"], c["completion"]["abs"]["p_not_higher"]),
                        "noninferior_unadjusted": non_inferior(c, NI_COLLISION, NI_COMPLETION)}
        for base in ("R6", best):
            for suffix in ("shuffled", "shifted"):
                ctrl = f"{base}_{suffix}"
                if ctrl in d and base in d:
                    rows[f"{ctrl}_vs_{base}"] = {"comparison": pair(d, ctrl, base)}
        if "R0" in d and "R1" in d:
            rows["R1_vs_R0"] = {"comparison": pair(d, "R1", "R0")}
        sec[cond] = rows
    res["secondary"] = sec
    out = HERE / "results" / "G1" / f"g1_{args.partition}.json"
    print(write_json(out, strip(res)))
    for cond, desc in res["descriptives"].items():
        figure(desc, HERE / "figures" / f"g1_{args.partition}_{cond}_frontier.png",
               f"G1 partition {args.partition}, condition {cond}")
    if "primary" in res:
        p = res["primary"]
        print(json.dumps({"p_holm": p["p_holm"], "P1": p["P1_R6_vs_R1"]["decision"],
                          "P2": p["P2_best_vs_R6_noninferiority"]["decision"], "P3": p["P3_best_vs_R1"]["decision"]},
                         indent=1, default=float))


if __name__ == "__main__":
    main()
