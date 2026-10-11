"""G2 analysis: adaptation (Kaplan–Meier, censored), predictive quality, false blame, selective feedback, closed-loop
behaviour; tuning objective and validation selection; confirmatory primary family.

Operational definitions (fixed in prereg/G2_ADDENDUM.md):
* adaptation (S1, S2; each switched identity): established iff at the end of episode 14 its memory p lies on the
  pre-switch side of the median p of the identities present; time = episodes from 15 until the end-of-episode p
  first lies on the post-switch side (1 = adapted by the end of the switch episode); never → censored at 15.
* per-encounter Brier: memory p at encounter start vs collision with that encounter, over *exposed* and collision
  encounters (where an outcome was observable); Oracle regret = Brier − Brier of the true-disposition rate.
* false blame (S5): final p of the patch-affine careful identity minus the mean final p of the other careful
  identities of the same lifetime.
* selective feedback: share of encounters that were avoided or unexposed; unnecessary avoidance = avoided
  encounters with careful identities / encounters with careful identities.

    python scripts/g2_analyze.py --partition P --power
    python scripts/g2_analyze.py --partition V --tuning       (Brier objective per configuration)
    python scripts/g2_analyze.py --partition X
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
from apm_select.stats import compare_systems, holm, non_inferior, strip  # noqa: E402
from apm_select.survival import km_bootstrap  # noqa: E402

RATES = json.loads((HERE.parents[0] / "F_behavioral_learning" / "configs" / "oracle_rates.json")
                   .read_text())["rates_by_disposition"]
SWITCH, HORIZON, EQUIV = 15, 15, 0.03


def load(path: Path) -> dict:
    data: dict = defaultdict(lambda: defaultdict(list))
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        data[r["label"]][r["scenario"]].append(r)
    return data


def adaptation_items(r: dict) -> list[tuple[float, bool]]:
    to = r["roles"]["to"]
    items = []
    for ident in r["roles"]["switched"]:
        def side(e, ident=ident):  # True = on the post-switch side of the median at the end of episode e
            snap = r["snapshots"][e]
            if ident not in snap:
                return None
            med = float(np.median([v[0] for v in snap.values()]))
            p = snap[ident][0]
            return p > med if to == "careless" else p < med
        pre = side(SWITCH - 1)
        if pre is None or pre:
            continue  # not established before the switch
        t = next((e - SWITCH + 1 for e in range(SWITCH, len(r["snapshots"])) if side(e)), None)
        items.append((float(t), True) if t is not None else (float(HORIZON), False))
    return items


def encounter_brier(rows: list[dict]) -> dict:
    sq, sq_or, n = 0.0, 0.0, 0
    bins = np.zeros((10, 3))
    for r in rows:
        for (_, _ident, _, klass, reliable, p0, disp, _) in r["encounters"]:
            if klass not in ("collision", "exposed") or p0 is None or not reliable or disp is None:
                continue
            y = 1.0 if klass == "collision" else 0.0
            sq += (p0 - y) ** 2
            sq_or += (RATES[disp] - y) ** 2
            n += 1
            b = min(9, int(p0 * 10))
            bins[b] += (1, p0, y)
    ece = float(sum(abs(c[1] - c[2]) for c in bins if c[0]) / max(1, n))
    return {"n": n, "brier": sq / max(1, n), "oracle_brier": sq_or / max(1, n),
            "regret": (sq - sq_or) / max(1, n), "ece": ece}


def false_blame(r: dict) -> float | None:
    final = r["snapshots"][-1]
    affine = r["roles"].get("affine")
    others = [v[0] for k, v in final.items() if k != affine and r["truth"][-1].get(k) == "careful"]
    if affine not in final or not others:
        return None
    return final[affine][0] - float(np.mean(others))


def selective(rows: list[dict]) -> dict:
    c = defaultdict(float)
    careful_total = careful_avoided = 0
    for r in rows:
        for k, v in r["memory_counts"].items():
            c[k] += v
        for (_, _ident, _, klass, _, _, disp, _) in r["encounters"]:
            if disp == "careful":
                careful_total += 1
                careful_avoided += klass == "avoided"
    total = sum(c[k] for k in ("collision", "exposed", "avoided", "unexposed"))
    return {"encounters": total, "no_evidence_share": (c["avoided"] + c["unexposed"]) / max(1, total),
            "avoided_share": c["avoided"] / max(1, total), "unreliable_share": c["unreliable"] / max(1, total),
            "unnecessary_avoidance": careful_avoided / max(1, careful_total)}


def boot_mean(vals: list[float], n_boot: int = 10_000) -> dict:
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return {"estimate": None}
    rng = np.random.default_rng(0)
    d = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return {"estimate": float(v.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "lower90": float(np.percentile(d, 5)), "upper90": float(np.percentile(d, 95)), "n": int(len(v)),
            "p_equiv": float(max(np.mean(d >= EQUIV), np.mean(d <= -EQUIV))), "se": float(d.std())}


def totals(rows: list[dict]) -> dict[int, dict]:
    return {(r["scenario"], r["index"]): r["totals"] for r in rows}


def closed_loop(a_rows: list[dict], b_rows: list[dict]) -> dict:
    ta, tb = totals(a_rows), totals(b_rows)
    keys = sorted(set(ta) & set(tb))
    arr = lambda t: {k: np.array([t[i][k] for i in keys], float) for k in t[keys[0]]}  # noqa: E731
    c = compare_systems(arr(ta), arr(tb))
    c["lifetimes"] = len(keys)
    return c


def summarise(data: dict) -> dict:
    out = {}
    for label, by_sc in data.items():
        rows = [r for rs in by_sc.values() for r in rs]
        s = {"brier": encounter_brier(rows), "selective": selective(rows),
             "state_size": max(r["state_size"] for r in rows)}
        for sc in ("S1", "S2"):
            if sc in by_sc:
                s[f"adaptation_{sc}"] = km_bootstrap([adaptation_items(r) for r in by_sc[sc]], HORIZON, 2000)
        if "S5" in by_sc:
            s["false_blame"] = boot_mean([false_blame(r) for r in by_sc["S5"]], 2000)
        for sc, rs in by_sc.items():
            t = {k: sum(r["totals"][k] for r in rs) for k in ("collisions", "minutes", "goals_completed",
                                                                "goals_issued", "careless_collisions")}
            s.setdefault("behaviour", {})[sc] = {"collision_rate": t["collisions"] / t["minutes"],
                                                 "completion": t["goals_completed"] / t["goals_issued"],
                                                 "careless_rate": t["careless_collisions"] / t["minutes"]}
        out[label] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/G2"))
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--tuning", action="store_true")
    ap.add_argument("--candidate", default=None, help="M* (from the validation selection)")
    ap.add_argument("--tag", default="g2", help="output file prefix")
    args = ap.parse_args()
    data = load(args.runs / args.partition / "lifetimes.jsonl")
    if args.tuning:
        by_cfg = defaultdict(list)
        for by_sc in data.values():
            for rs in by_sc.values():
                for r in rs:
                    by_cfg[r["config_id"]].append(r)
        table = {c: encounter_brier(rs) for c, rs in by_cfg.items()}
        write_json(HERE / "results" / "G2" / f"tuning_{args.partition}.json", table)
        print(json.dumps(table, indent=1))
        return
    if args.power:
        out = {}
        for label in data:
            if label != "M0_all":
                c = closed_loop([r for rs in data[label].values() for r in rs],
                                [r for rs in data["M0_all"].values() for r in rs])
                out[label] = {"lifetimes": c["lifetimes"], "se_collision_rel": c["collision_rate"]["rel"]["se"],
                              "se_completion_abs": c["completion"]["abs"]["se"],
                              "required_NI": int(np.ceil(c["lifetimes"] * max(
                                  (2.486 * c["collision_rate"]["rel"]["se"] / 0.10) ** 2,
                                  (2.486 * c["completion"]["abs"]["se"] / 0.02) ** 2)))}
        print(json.dumps(out, indent=1))
        return
    res: dict = {"partition": args.partition, "summary": summarise(data)}
    if args.candidate and args.partition == "X":
        cand, ref = args.candidate, "M0_all"
        s = res["summary"][cand]
        cl = closed_loop([r for rs in data[cand].values() for r in rs], [r for rs in data[ref].values() for r in rs])
        ni = non_inferior(cl, 0.10, 0.02)
        a1, a2 = s["adaptation_S1"], s["adaptation_S2"]
        fb = s["false_blame"]
        p = {"i": max(a1["p_median_above_6"], a2["p_median_above_6"]), "ii": ni["p"], "iii": fb["p_equiv"]}
        ph = holm(p)
        crit = {"i_adaptation": bool(a1["median"] <= 6 and a2["median"] <= 6 and a1["median_ci95"][1] <= 10
                                     and a2["median_ci95"][1] <= 10 and ph["i"] < 0.05),
                "ii_collision_noninferior": bool(ni["non_inferior"] and ph["ii"] < 0.05),
                "iii_false_blame": bool(fb["ci95"][0] > -EQUIV and fb["ci95"][1] < EQUIV and ph["iii"] < 0.05)}
        res["primary"] = {"candidate": cand, "reference": ref, "p": p, "p_holm": ph, "criteria": crit,
                          "closed_loop_vs_M0": cl, "noninferiority": ni, "all_pass": all(crit.values())}
    others = {}
    for label in data:
        if label != "M0_all" and "M0_all" in data and data[label]:
            others[label] = closed_loop([r for rs in data[label].values() for r in rs],
                                        [r for rs in data["M0_all"].values() for r in rs])
    res["closed_loop_vs_M0_all"] = others
    print(write_json(HERE / "results" / "G2" / f"{args.tag}_{args.partition}.json", strip(res)))
    brief = {k: {"brier": round(v["brier"]["brier"], 4), "regret": round(v["brier"]["regret"], 4),
                 "S1": (v.get("adaptation_S1") or {}).get("median"), "S2": (v.get("adaptation_S2") or {}).get("median"),
                 "no_evidence": round(v["selective"]["no_evidence_share"], 3)} for k, v in res["summary"].items()}
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
