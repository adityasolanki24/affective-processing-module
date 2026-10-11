"""E3 analysis: criteria (i)–(iv) on B1 memory (B2 alongside), secondary metrics, pilot sizing.

Operational definitions (fixed in prereg/E3_ADDENDUM.md before the confirmatory run):
* ranking AUROC: per lifetime, memory p (collision per encounter) of the 6 identities at the end of episode 29 vs
  true disposition (careless = positive); mean over lifetimes, bootstrap CI over lifetimes.
* adaptation delay (S1/S2, per switched identity): the identity must be on its pre-switch side of the population
  median at the end of episode 14 (else "not established", counted and excluded); delay = number of episodes from
  15 until the end-of-episode snapshot first lies on the post-switch side; never → censored at 16. Median over
  identity-lifetimes, bootstrap CI over lifetimes.
* false attribution (S5): per lifetime, final p of the patch-affine careful identity minus the mean final p of the
  other careful identities of the same lifetime that were met at least once; mean, bootstrap CI.
* usefulness (S3): careless-identity collisions per minute, B1 vs B0 (paired ratio of totals), and completion
  difference.

    python scripts/e3_analyze.py --partition P --power
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
from e3_run import EPISODES, SWITCH  # noqa: E402

from apm.learning.evaluation import auroc  # noqa: E402
from apm_research.provenance import write_json  # noqa: E402
from apm_research.stats import paired_lifetimes  # noqa: E402

N_BOOT = 10_000
CENSORED = EPISODES - SWITCH + 1


def boot_mean(values: np.ndarray, stat=np.mean, seed: int = 0) -> dict:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = np.array([stat(values[rng.integers(0, len(values), len(values))]) for _ in range(N_BOOT)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(stat(values)), "ci95": [float(lo), float(hi)], "se": float(draws.std()),
            "n": int(len(values))}


def boot_clustered(groups: list[list[float]], stat=np.median, seed: int = 0) -> dict:
    """Statistic over pooled items, resampling lifetimes (each a list of items)."""
    rng = np.random.default_rng(seed)
    flat = np.concatenate([np.asarray(g, float) for g in groups]) if groups else np.array([])
    draws = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        draws.append(stat(np.concatenate([np.asarray(groups[i], float) for i in pick])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(stat(flat)), "ci95": [float(lo), float(hi)], "se": float(np.std(draws)),
            "lifetimes": len(groups), "items": int(len(flat))}


def snapshots(r: dict) -> dict[int, dict[str, tuple]]:
    out: dict[int, dict[str, tuple]] = defaultdict(dict)
    for e, name, disp, p, conf, naive in r["snapshots"]:
        out[e][name] = (disp, p, conf, naive)
    return out


def ranking_auroc(r: dict, key: int = 1, episode: int = EPISODES - 1) -> float:
    snap = snapshots(r)[episode]
    return auroc([v[key] for v in snap.values()], [v[0] == "careless" for v in snap.values()])


def adaptation(r: dict, key: int = 1) -> tuple[list[float], int]:
    snap = snapshots(r)
    to_careless = r["roles"]["to"] == "careless"
    delays, not_established = [], 0
    for name in r["roles"]["switched"]:
        def side(e, name=name):  # True = on the post-switch side of the population median
            med = np.median([v[key] for v in snap[e].values()])
            p = snap[e][name][key]
            return p > med if to_careless else p < med
        if side(SWITCH - 1):
            not_established += 1
            continue
        delay = next((e - SWITCH + 1 for e in range(SWITCH, EPISODES) if side(e)), CENSORED)
        delays.append(float(delay))
    return delays, not_established


def false_attribution(r: dict, key: int = 1) -> float | None:
    final = snapshots(r)[EPISODES - 1]
    affine = r["roles"]["affine"]
    controls = [v[key] for n, v in final.items() if n != affine and v[0] == "careful" and v[2] > 0]
    if not controls:
        return None
    return float(final[affine][key] - np.mean(controls))


def totals(r: dict, collisions_key: str = "careless_collisions") -> dict:
    rows = r["rows"]
    return {"collisions": sum(x[collisions_key] for x in rows), "minutes": sum(x["minutes"] for x in rows),
            "goals_completed": sum(x["goals_completed"] for x in rows),
            "goals_issued": sum(x["goals_issued"] for x in rows), "idle": sum(x["idle"] for x in rows),
            "periods": sum(x["periods"] for x in rows)}


def paired(a: list[dict], b: list[dict], key: str = "careless_collisions") -> dict:
    ia, ib = {r["index"]: r for r in a}, {r["index"]: r for r in b}
    common = sorted(set(ia) & set(ib))
    ta = [totals(ia[i], key) for i in common]
    tb = [totals(ib[i], key) for i in common]
    return paired_lifetimes({k: np.array([t[k] for t in ta]) for k in ta[0]},
                            {k: np.array([t[k] for t in tb]) for k in tb[0]})


def excess_avoidance(r: dict) -> float:
    near = [(disp, a) for _, disp, a in r["near"] if disp == "careful"]
    return float(np.mean([a != 0 for _, a in near])) if near else float("nan")


def regret(rs: list[dict]) -> dict:
    """Brier of memory per-encounter predictions vs an oracle that knows each encounter's true disposition and the
    true per-encounter collision rate of that disposition under the same system and scenario (pooled over
    lifetimes; behaviour-dependent rates make an exploration-based rate the wrong reference here)."""
    enc = [x for r in rs for x in r["encounters"] if x[2] is not None]
    if not enc:
        return {}
    y = np.array([x[6] for x in enc], float)
    p = np.array([x[4] for x in enc], float)
    disp = np.array([x[2] for x in enc])
    o = np.zeros_like(y)
    for d in np.unique(disp):
        o[disp == d] = y[disp == d].mean()
    return {"memory_brier": float(np.mean((p - y) ** 2)), "oracle_brier": float(np.mean((o - y) ** 2)),
            "regret": float(np.mean((p - y) ** 2) - np.mean((o - y) ** 2)), "encounters": int(len(y)),
            "true_rate": {str(d): float(y[disp == d].mean()) for d in np.unique(disp)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/E3"))
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    data: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(args.runs / args.partition / "lifetimes.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            data[(r["scenario"], r["system"])].append(r)
    rates = json.loads((HERE / "configs" / "oracle_rates.json").read_text())["rates_by_disposition"]
    memory_ok = all(r["observer_equals_policy_memory"] is not False for rs in data.values() for r in rs)
    res: dict = {"partition": args.partition, "observer_memory_equals_policy_memory": memory_ok,
                 "lifetimes": {f"{k[0]}|{k[1]}": len(v) for k, v in data.items()}}
    systems = ("B1", "B2", "B1eps", "B0", "Oracle")

    # (i) ranking AUROC, S3 (B1 primary) and S4 (identity errors, secondary); naive memory reference
    res["i_ranking_auroc"] = {}
    for sc in ("S3", "S4"):
        for s in systems:
            if (sc, s) in data:
                vals = [x for x in (ranking_auroc(r) for r in data[(sc, s)]) if not np.isnan(x)]
                res["i_ranking_auroc"][f"{sc}|{s}"] = boot_mean(vals)
        if (sc, "B1") in data:
            vals = [x for x in (ranking_auroc(r, key=3) for r in data[(sc, "B1")]) if not np.isnan(x)]
            res["i_ranking_auroc"][f"{sc}|naive_under_B1"] = boot_mean(vals)
    # (ii) adaptation delay
    res["ii_adaptation_delay"] = {}
    for sc in ("S1", "S2"):
        for s in systems:
            if (sc, s) in data:
                groups, ne = [], 0
                for r in data[(sc, s)]:
                    d, n = adaptation(r)
                    groups.append(d)
                    ne += n
                groups = [g for g in groups if g]
                entry = boot_clustered(groups) if groups else {"estimate": None}
                entry["not_established"] = ne
                entry["censored_fraction"] = float(np.mean([x == CENSORED for g in groups for x in g])) \
                    if groups else None
                res["ii_adaptation_delay"][f"{sc}|{s}"] = entry
    # (iii) false attribution
    res["iii_false_attribution"] = {}
    for s in systems:
        if ("S5", s) in data:
            vals = [x for x in (false_attribution(r) for r in data[("S5", s)]) if x is not None]
            res["iii_false_attribution"][f"S5|{s}"] = boot_mean(vals)
    if ("S5", "B1") in data:
        vals = [x for x in (false_attribution(r, key=3) for r in data[("S5", "B1")]) if x is not None]
        res["iii_false_attribution"]["S5|naive_under_B1"] = boot_mean(vals)
        # does the affine identity actually collide more per encounter? (truth, not memory)
        enc = [(x[1] == r["roles"]["affine"], x[6]) for r in data[("S5", "B1")] for x in r["encounters"]
               if x[2] == "careful"]
        res["iii_false_attribution"]["S5|B1_true_collision_rate_per_encounter"] = {
            "affine": float(np.mean([c for a, c in enc if a])), "other_careful": float(np.mean([c for a, c in enc
                                                                                               if not a]))}
    # (iv) usefulness, S3
    res["iv_usefulness"] = {}
    for a, b in (("B1", "B0"), ("B2", "B0"), ("Oracle", "B0"), ("B1", "Oracle"), ("B2", "B1")):
        if ("S3", a) in data and ("S3", b) in data:
            res["iv_usefulness"][f"S3|{a}_vs_{b}"] = paired(data[("S3", a)], data[("S3", b)])
    # secondary: regret, overgeneralisation, excess avoidance, censoring
    res["predictive_regret"] = {}
    for (sc, s), rs in sorted(data.items()):
        if s != "Oracle":
            res["predictive_regret"][f"{sc}|{s}"] = regret(rs)
    res["overgeneralisation_S7"] = {}
    for s in systems:
        if ("S7", s) in data:
            vals = []
            for r in data[("S7", s)]:
                first = next((x for x in r["encounters"] if x[1] == r["roles"]["newcomer"]), None)
                if first is not None:
                    vals.append(first[4] - rates["careful"])
            res["overgeneralisation_S7"][s] = boot_mean(vals) if vals else None
    res["excess_avoidance_nonGO_near_careful"] = {}
    for sc in ("S3",):
        for s in systems:
            if (sc, s) in data:
                vals = [x for x in (excess_avoidance(r) for r in data[(sc, s)]) if not np.isnan(x)]
                res["excess_avoidance_nonGO_near_careful"][f"{sc}|{s}"] = boot_mean(vals)
    res["censoring_S6"] = {}
    for s in ("B0", "B1", "B1eps", "B2", "Oracle"):
        if ("S6", s) in data:
            rs = data[("S6", s)]
            final = [snapshots(r)[EPISODES - 1]["ped-0"] for r in rs]
            enc = [x[6] for r in rs for x in r["encounters"] if x[1] == "ped-0"]
            res["censoring_S6"][s] = {"final_p_avoided_identity": boot_mean([f[1] for f in final]),
                                      "true_rate_careless": rates["careless"],
                                      "observed_collisions_per_encounter": float(np.mean(enc)) if enc else None}

    if args.power:
        n = {k: len(v) for k, v in data.items()}
        sizing = {}
        i = res["i_ranking_auroc"].get("S3|B1")
        if i:
            sizing["i"] = int(np.ceil(i["n"] * (2.8 * i["se"] / 0.10) ** 2))
        fa = res["iii_false_attribution"].get("S5|B1")
        if fa:
            room = max(1e-3, 0.03 - abs(fa["estimate"]))
            sizing["iii"] = int(np.ceil(fa["n"] * (1.96 * fa["se"] / room) ** 2))
        u = res["iv_usefulness"].get("S3|B1_vs_B0")
        if u:
            se = (u["collision_rate_relative"]["ci95"][1] - u["collision_rate_relative"]["ci95"][0]) / 3.92
            sizing["iv"] = int(np.ceil(n[("S3", "B1")] * (2.8 * se / 0.20) ** 2))
        print(json.dumps({"lifetimes": res["lifetimes"], "required": sizing}, indent=1))
        return
    i = res["i_ranking_auroc"].get("S3|B1", {})
    s1, s2 = res["ii_adaptation_delay"].get("S1|B1", {}), res["ii_adaptation_delay"].get("S2|B1", {})
    fa = res["iii_false_attribution"].get("S5|B1", {})
    u = res["iv_usefulness"].get("S3|B1_vs_B0", {})
    crit = {
        "i_ranking": bool(i and i["estimate"] >= 0.80 and i["ci95"][0] >= 0.70),
        "ii_adaptation": bool(s1.get("estimate") is not None and s2.get("estimate") is not None
                              and s1["estimate"] <= 6 and s2["estimate"] <= 6),
        "iii_false_attribution": bool(fa and -0.03 < fa["ci95"][0] and fa["ci95"][1] < 0.03),
        "iv_usefulness": bool(u and u["collision_rate_relative"]["estimate"] <= -0.20
                              and u["collision_rate_relative"]["ci95"][1] < 0.0
                              and u["completion_difference"]["ci95"][0] >= -0.03)}
    names = {"i_ranking": "encounter statistics only (no reliable individual ranking)",
             "ii_adaptation": "encounter statistics only (no timely revision)",
             "iii_false_attribution": "misleading statistics (false attribution)",
             "iv_usefulness": "encounter statistics only (memory-informed control not useful)"}
    res["criteria"] = crit
    res["verdict"] = "adaptive dispositions" if all(crit.values()) else \
        "; ".join(names[k] for k, v in crit.items() if not v)
    out = args.out or HERE / "results" / "E3" / f"e3_{args.partition}.json"
    print(write_json(out, res))
    print(json.dumps({"criteria": crit, "verdict": res["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
