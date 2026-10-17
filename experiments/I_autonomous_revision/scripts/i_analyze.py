"""Programme I analysis (definitions and criteria: MASTER_PREREGISTRATION.md §5–§6).

    python experiments/I_autonomous_revision/scripts/i_analyze.py --experiment I1 --runs <lifetimes.jsonl ...> \
        --out experiments/I_autonomous_revision/results/I1/I1_X.json

Per system:
- calibration (S3);
- adaptation (S1, S2, S10), both system-calibrated (primary) and G2's median form (secondary);
- discrimination (S3 ranking AUROC);
- false blame (S5), raw and normalised by the system's careless–careful gap;
- evidence (censored share), probes, keep-aways;
- per-family completion and avoidable collisions.

Paired comparisons use G's bootstrap (`apm_select.stats.compare_systems`, 10,000 resamples over lifetimes). The
criteria of the experiment are then evaluated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i_metrics import (  # noqa: E402
    adaptation,
    adaptation_cal,
    boot_mean,
    calibration,
    censored_share,
    false_blame,
    load,
    ranking_auroc,
)

from apm_select.stats import compare_systems, holm, non_inferior, p_beyond, strip  # noqa: E402

FAMILIES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10")
ADAPT = ("S1", "S2", "S10")
NI_COLLISION, NI_COMPLETION = 0.10, 0.02  # G's margins
ROLES = {
    "I1": {"candidate": "R_dense", "reference": "A0"},
    "I2": {"candidate": "KA_R_voi", "reference": "KA_A0", "no_explore": "KA_R", "random": "KA_R_rand"},
}


def totals(rows: list[dict]) -> dict:
    return {(r["scenario"], r["index"]): r["totals"] for r in rows}


def paired(a_rows: list[dict], b_rows: list[dict]) -> dict:
    ta, tb = totals(a_rows), totals(b_rows)
    keys = sorted(set(ta) & set(tb))
    arr = lambda t: {k: np.array([t[i][k] for i in keys], float) for k in t[keys[0]]}  # noqa: E731
    c = compare_systems(arr(ta), arr(tb))
    c["lifetimes"] = len(keys)
    return c


def per_system(by_sc: dict) -> dict:
    cal = calibration(by_sc["S3"]) if by_sc.get("S3") else None
    out: dict = {"calibration": cal, "families": {}}
    for sc, rows in sorted(by_sc.items()):
        t = [r["totals"] for r in rows]
        out["families"][sc] = {
            "lifetimes": len(rows),
            "completion": sum(x["goals_completed"] for x in t) / max(1.0, sum(x["goals_issued"] for x in t)),
            "avoidable_per_min": sum(x["avoidable_collisions"] for x in t) / sum(x["minutes"] for x in t),
            "careless_per_min": sum(x["careless_collisions"] for x in t) / sum(x["minutes"] for x in t),
            "censored_share": censored_share(rows),
            "probes_per_lifetime": float(np.mean([r["probes"] for r in rows])),
            "keepaways_per_lifetime": float(np.mean([r.get("keepaways", 0) for r in rows])),
        }
    for sc in ADAPT:
        if by_sc.get(sc) and cal:
            out[f"adaptation_{sc}"] = adaptation_cal(by_sc[sc], cal)
            out[f"adaptation_{sc}_median_form"] = adaptation(by_sc[sc])
    if by_sc.get("S3"):
        out["discrimination_S3"] = boot_mean([ranking_auroc(r) for r in by_sc["S3"]])
    if by_sc.get("S5") and cal:
        fb = [false_blame(r) for r in by_sc["S5"]]
        out["false_blame"] = boot_mean(fb)
        gap = cal["gap"]
        out["false_blame_normalised"] = boot_mean([None if x is None or abs(gap) < 1e-12 else x / abs(gap)
                                                   for x in fb])
    return out


def adaptation_ok(a: dict | None) -> bool:
    if not a or a.get("median") is None:
        return False
    return bool(a["median"] <= 6 and a["median_ci95"][1] <= 10)


def system_criteria(s: dict, cmp_ref: dict) -> dict:
    """Pass criteria C1–C5 for one system against its reference (§6)."""
    disc, fbn = s.get("discrimination_S3") or {}, s.get("false_blame_normalised") or {}
    ni = non_inferior(cmp_ref, NI_COLLISION, NI_COMPLETION, metric="avoidable_collision_rate")
    p_col = p_beyond(cmp_ref["avoidable_collision_rate"]["rel"], NI_COLLISION, "below")
    p_comp = p_beyond(cmp_ref["completion"]["abs"], -NI_COMPLETION, "above")
    adj = holm({"C3": p_col, "C4": p_comp})
    c = {
        "C1_adaptation": {sc: adaptation_ok(s.get(f"adaptation_{sc}")) for sc in ADAPT},
        "C2_discrimination": bool(disc.get("estimate") is not None and disc["estimate"] >= 0.80
                                  and disc["ci95"][0] >= 0.70),
        "C3_safety_NI": {"upper95": ni["collision_upper95"], "p": p_col, "p_holm": adj["C3"],
                         "pass": bool(adj["C3"] < 0.05)},
        "C4_completion_NI": {"lower95": ni["completion_lower95"], "p": p_comp, "p_holm": adj["C4"],
                             "pass": bool(adj["C4"] < 0.05)},
        "C5_false_blame": bool(fbn.get("estimate") is not None and abs(fbn["estimate"]) < 0.25
                               and max(abs(fbn["ci95"][0]), abs(fbn["ci95"][1])) < 0.5),
    }
    c["pass"] = bool(all(c["C1_adaptation"].values()) and c["C2_discrimination"] and c["C3_safety_NI"]["pass"]
                     and c["C4_completion_NI"]["pass"] and c["C5_false_blame"])
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=sorted(ROLES))
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = load(a.runs)
    roles = ROLES[a.experiment]
    res: dict = {"experiment": a.experiment, "roles": roles, "systems": {}, "comparisons": {}}
    for name, by_sc in data.items():
        res["systems"][name] = per_system(by_sc)
    allrows = {n: [r for rs in by_sc.values() for r in rs] for n, by_sc in data.items()}
    ref = roles["reference"]
    for name in data:
        if name != ref:
            res["comparisons"][f"{name}_vs_{ref}"] = paired(allrows[name], allrows[ref])
    cand = roles["candidate"]
    res["criteria"] = {"candidate": system_criteria(res["systems"][cand], res["comparisons"][f"{cand}_vs_{ref}"])}
    for name in data:  # the same criteria for every other system, as secondary information
        if name not in (cand, ref) and "Oracle" not in name:
            res["criteria"][name] = system_criteria(res["systems"][name], res["comparisons"][f"{name}_vs_{ref}"])
    if a.experiment == "I2":  # primary family P1–P3 (§6.2), Holm
        s10 = {n: data[n]["S10"] for n in data}
        c_none = paired(s10[cand], s10[roles["no_explore"]])
        c_rand = paired(s10[cand], s10[roles["random"]])
        c_all = paired(allrows[cand], allrows[roles["no_explore"]])
        p = {"P1": p_beyond(c_none["completion"]["abs"], 0.0, "above"),
             "P2": p_beyond(c_all["avoidable_collision_rate"]["rel"], NI_COLLISION, "below"),
             "P3": p_beyond(c_rand["completion"]["abs"], 0.0, "above")}
        adj = holm(p)
        res["primary_I2"] = {"p": p, "p_holm": adj, "pass": {k: bool(v < 0.05) for k, v in adj.items()},
                             "S10_completion_vs_none": c_none["completion"]["abs"],
                             "S10_completion_vs_random": c_rand["completion"]["abs"],
                             "avoidable_vs_none": c_all["avoidable_collision_rate"]["rel"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(strip(res), indent=1, default=float), encoding="utf-8")
    for name, s in res["systems"].items():
        ad = {sc: (s.get(f"adaptation_{sc}") or {}).get("median") for sc in ADAPT}
        print(f"{name:12} adapt {ad} disc {(s.get('discrimination_S3') or {}).get('estimate')} "
              f"fb {(s.get('false_blame_normalised') or {}).get('estimate')}")
    print(json.dumps(strip(res["criteria"]), indent=1, default=float)[:3000])
    if "primary_I2" in res:
        print(json.dumps(strip(res["primary_I2"]), indent=1, default=float))


if __name__ == "__main__":
    main()
