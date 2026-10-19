"""Remediation corrected comparisons: matched supervision, fall-aware outcomes, identity controls (frozen protocol:
REMEDIATION_PROTOCOL.md, "Corrected comparisons").

    python experiments/M_release/scripts/r_analyze.py --runs <RY dirs or jsonl> \
        --tasks nova_carter=<tasks.json> ... --out results/remediation/corrected.json

Lifetimes: experiment R, partition Y only. Pairs are matched on (history, index). Per lifetime:
- cost_planned: the mean realised cost over all planned encounters, where each encounter lost to a fall counts
  as a failed encounter (the 45 s approach budget plus the retry time). Falls carry no extra penalty here;
- cost_survived: the mean over the encounters that took place;
- fell, complete (all planned encounters), contacts, peak contact impulse, task success, approaches completed,
  supervisor vetoes.

Tests:
- one-sided paired sign-flip permutation tests (10 000, seed 20261019), Holm within robot and family, 95%
  bootstrap CIs;
- fall regression: one-sided Fisher exact test on lifetimes with a fall (DT more than the comparator).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from m6_analyze import APPROACH_BUDGET_S, boot_ci, enc_cost, holm, load, perm_p

N_PLANNED = 24
MAIN_VS = ("MONOTONE", "FIXED", "NEUTRAL", "LOOKUP", "NOMEM")
ID_VS = ("POOLED", "SWAPPED", "SHUFFLED")


def fisher_greater(a_fall: int, a_n: int, b_fall: int, b_n: int) -> float:
    """P(DT has at least this many fall lifetimes) under the hypergeometric null; one-sided, DT > comparator."""
    from math import comb
    total, falls = a_n + b_n, a_fall + b_fall
    denom = comb(total, falls)
    return float(sum(comb(a_n, k) * comb(b_n, falls - k) for k in range(a_fall, min(a_n, falls) + 1)) / denom) \
        if denom else 1.0


def lifetime_measures(rec: dict, task: dict) -> dict:
    rows = rec["rows"]
    costs = [enc_cost(r, task) for r in rows]
    lost = max(0, N_PLANNED - len(rows))
    lost_cost = APPROACH_BUDGET_S + task["retry_s"]
    fell = any(r["fell"] for r in rows)
    imp = [r.get("impulse_max", r.get("max_push")) for r in rows]
    return {"cost_planned": (sum(costs) + lost * lost_cost) / N_PLANNED,
            "cost_survived": float(np.mean(costs)) if costs else None,
            "fell": fell, "complete": len(rows) >= N_PLANNED and not fell,
            "contacts": sum(r["contacts"] for r in rows) / max(1, len(rows)),
            "impulse_peak": max((float(x) for x in imp if x is not None), default=0.0),
            "success": float(np.mean([r["d_stop"] <= task["d_task"] + 0.05 for r in rows])) if rows else 0.0,
            "approaches": sum(1 for r in rows if r.get("standoff", 0) == 0 and r.get("arrived")),
            "vetoes": rec.get("vetoes", 0)}


def identity_measures(rec: dict, task: dict) -> dict:
    rows = rec["rows"]
    b = [r for r in rows if r["entity"] == "B1"][2:]
    calm = [r for r in rows if r["entity"] != "B1"]
    return {"b1_contacts": float(np.mean([r["contacts"] for r in b])) if b else None,
            "calm_cost": float(np.mean([enc_cost(r, task) for r in calm])) if calm else None}


def pairs_of(recs: list[dict], robot: str, histories: tuple[str, ...], a: str, b: str) -> list[tuple[dict, dict]]:
    idx: dict = {}
    for r in recs:
        if r["robot"] != robot or r["history"] not in histories:
            continue
        name = r["system"].split(":")[0]
        if name in (a, b):
            idx.setdefault((r["history"], r["index"]), {})[name] = r
    return [(v[a], v[b]) for v in idx.values() if a in v and b in v]


def compare(pairs, task, key: str, direction: str = "less", fn=lifetime_measures) -> dict:
    d = np.array([fn(x, task)[key] - fn(y, task)[key] for x, y in pairs
                  if fn(x, task)[key] is not None and fn(y, task)[key] is not None], float)
    return {"n_pairs": int(len(d)), "mean_diff_DT_minus_other": float(d.mean()) if len(d) else None,
            "ci95": boot_ci(d), "p_one_sided": perm_p(d, direction)}


def summary(recs: list[dict], task: dict) -> dict:
    m = [lifetime_measures(r, task) for r in recs]
    if not m:
        return {}
    return {"lifetimes": len(m), "fall_lifetimes": sum(x["fell"] for x in m),
            "complete_rate": float(np.mean([x["complete"] for x in m])),
            "cost_planned": float(np.mean([x["cost_planned"] for x in m])),
            "cost_survived": float(np.mean([x["cost_survived"] for x in m if x["cost_survived"] is not None])),
            "contacts_per_encounter": float(np.mean([x["contacts"] for x in m])),
            "impulse_peak_max": float(max(x["impulse_peak"] for x in m)),
            "success": float(np.mean([x["success"] for x in m])),
            "approaches": float(np.mean([x["approaches"] for x in m])), "vetoes": int(sum(x["vetoes"] for x in m))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    recs = [r for r in load(a.runs) if r.get("experiment") == "R" and r.get("partition") == "Y"]
    tasks = {kv.split("=")[0]: json.loads(Path(kv.split("=")[1]).read_text())["tasks"]["handoff"] for kv in a.tasks}
    res: dict = {"lifetimes": len(recs), "robots": {}}
    for robot in sorted({r["robot"] for r in recs} & set(tasks)):
        task = tasks[robot]
        rr = [r for r in recs if r["robot"] == robot]
        out: dict = {"systems": {}}
        for name in sorted({r["system"].split(":")[0] for r in rr}):
            for fam, hist in (("main", ("change_B", "late_B")), ("identity", ("risky_B",))):
                sel = [r for r in rr if r["system"].split(":")[0] == name and r["history"] in hist]
                if sel:
                    out["systems"][f"{fam}/{name}"] = summary(sel, task)
        tests, ps = {}, {}
        for other in MAIN_VS:
            pr = pairs_of(rr, robot, ("change_B", "late_B"), "DT", other)
            if not pr:
                continue
            t = compare(pr, task, "cost_planned")
            t["cost_survived"] = compare(pr, task, "cost_survived")
            fa = sum(lifetime_measures(x, task)["fell"] for x, _ in pr)
            fb = sum(lifetime_measures(y, task)["fell"] for _, y in pr)
            t["falls"] = {"DT": fa, "other": fb, "n": len(pr),
                          "p_DT_more_falls": fisher_greater(fa, len(pr), fb, len(pr))}
            tests[f"main/DT_vs_{other}"], ps[f"main/DT_vs_{other}"] = t, t["p_one_sided"]
        sig = holm(ps)
        for k, t in tests.items():
            t["holm_pass"] = sig[k]
            t["better_overall"] = bool(sig[k] and t["falls"]["p_DT_more_falls"] > 0.05)
        id_tests, id_ps = {}, {}
        for other in ID_VS:
            pr = pairs_of(rr, robot, ("risky_B",), "DT", other)
            if not pr:
                continue
            t = compare(pr, task, "cost_planned")
            t["b1_contacts"] = compare(pr, task, "b1_contacts", fn=identity_measures)
            t["calm_cost"] = compare(pr, task, "calm_cost", fn=identity_measures)
            id_tests[f"identity/DT_vs_{other}"], id_ps[f"identity/DT_vs_{other}"] = t, t["p_one_sided"]
        id_sig = holm(id_ps)
        for k, t in id_tests.items():
            t["holm_pass"] = id_sig[k]
        pooled, swapped = id_tests.get("identity/DT_vs_POOLED"), id_tests.get("identity/DT_vs_SWAPPED")
        supported = bool(pooled and swapped and pooled["holm_pass"] and swapped["holm_pass"]
                         and (pooled["b1_contacts"]["mean_diff_DT_minus_other"] or 0) <= 0)
        out["tests"], out["identity_tests"] = tests, id_tests
        out["identity_specific_supported"] = supported
        res["robots"][robot] = out
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for robot, o in res["robots"].items():
        print(robot)
        for k, t in {**o["tests"], **o["identity_tests"]}.items():
            md = t["mean_diff_DT_minus_other"]
            print(f"  {k:28} n={t['n_pairs']:2} cost diff={md if md is None else round(md, 2)} "
                  f"p={t['p_one_sided']:.4f} holm={t['holm_pass']}" + (f" falls={t['falls']}" if "falls" in t else ""))
        print("  identity-specific supported:", o["identity_specific_supported"])


if __name__ == "__main__":
    main()
