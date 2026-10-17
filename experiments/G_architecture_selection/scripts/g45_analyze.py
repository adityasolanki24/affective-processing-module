"""G4/G5 analysis (prereg/G45_ADDENDUM.md). Selections use training/validation records only.

G4 primary (Holm over 2 × shortlist p-values): for each architecture, its best method (highest mean best-validation
J over seeds) vs A0 under the superiority rule, two-level bootstrap (lifetimes, then training seeds).
G5 primary (Holm over 3): best architecture's best method vs its PPO-from-scratch (superiority: p_A, p_B), and the
best ordinary-GRU cell (A2/A3) non-inferior to A0 (p_NI).
Secondary: every cell vs A0 and vs its PPO cell; learning-curve area; interaction counts; action distributions and
action entropy (collapse); seed variance; A1 vs A0.

    python scripts/g45_analyze.py --partition X
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
from apm_select.provenance import write_json  # noqa: E402
from apm_select.stats import compare_systems, e2_rule, holm, non_inferior, strip  # noqa: E402

sys.path.insert(0, str(HERE / "scripts"))


def load(path: Path) -> dict:
    rows: dict = defaultdict(lambda: defaultdict(dict))
    acts: dict = defaultdict(lambda: np.zeros(4))
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[r["label"]][r["seed"]][r["index"]] = r["result"]
        acts[r["label"]] += np.array(r["actions"])
    out = {}
    for label, by_seed in rows.items():
        seeds = sorted(by_seed)
        idx = sorted(set.intersection(*(set(v) for v in by_seed.values())))
        keys = by_seed[seeds[0]][idx[0]].keys()
        out[label] = {"arrays": {k: np.array([[by_seed[s][i][k] for i in idx] for s in seeds]) for k in keys},
                      "index": idx, "seeds": seeds, "actions": acts[label]}
    return out


def pair(data, a, b) -> dict:
    keep = sorted(set(data[a]["index"]) & set(data[b]["index"]))
    sel = lambda s: {k: v[:, [data[s]["index"].index(i) for i in keep]] for k, v in data[s]["arrays"].items()}  # noqa
    return compare_systems(sel(a), sel(b))


def describe(d: dict) -> dict:
    t = {k: float(v.sum()) for k, v in d["arrays"].items()}
    frac = d["actions"] / d["actions"].sum()
    return {"seeds": len(d["seeds"]), "lifetimes": len(d["index"]),
            "collision_rate": t["collisions"] / t["minutes"], "completion": t["goals_completed"] / t["goals_issued"],
            "severe_rate": t["severe"] / t["minutes"], "idle_fraction": t["idle"] / t["periods"],
            "J_mean": t["J"] / d["arrays"]["J"].size, "J_per_seed": d["arrays"]["J"].mean(1).round(3).tolist(),
            "action_fractions": frac.round(4).tolist(),
            "action_entropy_bits": float(-(frac[frac > 0] * np.log2(frac[frac > 0])).sum())}


def training_records(models: Path) -> dict:
    rec = defaultdict(list)
    for f in models.glob("*_s?.json"):
        r = json.loads(f.read_text())
        rec[f"{r['arch']}_{r['method']}"].append(r)
    return {k: {"val_J_mean": float(np.mean([r["best"]["J"] for r in v])),
                "val_J_sd": float(np.std([r["best"]["J"] for r in v])),
                "curve_area_mean": float(np.mean([r["curve_area_per_decision"] for r in v])),
                "decisions": int(np.mean([r["decisions_used"] for r in v])), "seeds": len(v),
                "params": v[0]["params"], "wall_s_mean": float(np.mean([r["wall_s"] for r in v]))}
            for k, v in rec.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/G45"))
    args = ap.parse_args()
    data = load(args.runs / f"eval_{args.partition}" / "lifetimes.jsonl")
    rec = training_records(args.runs / "models")
    archs = sorted({k.split("_")[0] for k in rec})
    best_cell = {a: max((k for k in rec if k.startswith(a + "_")), key=lambda k: rec[k]["val_J_mean"]) for a in archs}
    pv, comps = {}, {}
    for cell in best_cell.values():
        c = pair(data, cell, "A0")
        comps[cell] = c
        pv[f"{cell}:A"] = c["collision_rate"]["rel"]["p_not_lower"]
        pv[f"{cell}:B"] = c["completion"]["abs"]["p_not_higher"]
    ph = holm(pv)
    g4 = {cell: {"comparison": comps[cell], "decision": e2_rule(comps[cell], ph[f"{cell}:A"], ph[f"{cell}:B"])}
          for cell in comps}
    best_arch = max(archs, key=lambda a: rec[best_cell[a]]["val_J_mean"])
    gru_cells = [k for k in rec if k.split("_")[0] in ("A2", "A3")]
    best_gru = max(gru_cells, key=lambda k: rec[k]["val_J_mean"]) if gru_cells else None
    g5 = {"best_architecture": best_arch, "best_cell": best_cell[best_arch], "best_gru": best_gru}
    p5 = {}
    ppo_cell = f"{best_arch}_ppo"
    if best_cell[best_arch] != ppo_cell and ppo_cell in data:
        c = pair(data, best_cell[best_arch], ppo_cell)
        g5["method_vs_ppo"] = c
        p5["methodA"], p5["methodB"] = c["collision_rate"]["rel"]["p_not_lower"], c["completion"]["abs"]["p_not_higher"]
    if best_gru:
        c = pair(data, best_gru, "A0")
        ni = non_inferior(c, 0.10, 0.02)
        g5["best_gru_vs_A0"] = {"comparison": c, "noninferiority": ni}
        p5["gruNI"] = ni["p"]
    p5h = holm(p5) if p5 else {}
    if "method_vs_ppo" in g5:
        g5["method_vs_ppo_decision"] = e2_rule(g5["method_vs_ppo"], p5h["methodA"], p5h["methodB"])
    if best_gru:
        g5["best_gru_noninferior_after_holm"] = bool(g5["best_gru_vs_A0"]["noninferiority"]["non_inferior"]
                                                     and p5h["gruNI"] < 0.05)
    g5["p_holm"] = p5h
    secondary = {f"{s}_vs_A0": pair(data, s, "A0") for s in data if s != "A0"}
    for k in rec:
        arch = k.split("_")[0]
        if not k.endswith("_ppo") and f"{arch}_ppo" in data and k in data:
            secondary[f"{k}_vs_{arch}_ppo"] = pair(data, k, f"{arch}_ppo")
    res = {"partition": args.partition, "training": rec, "best_cell_per_architecture": best_cell,
           "descriptives": {s: describe(d) for s, d in data.items()},
           "G4_primary": {"p": pv, "p_holm": ph, "tests": g4}, "G5_primary": g5, "secondary": secondary}
    print(write_json(HERE / "results" / "G45" / f"g45_{args.partition}.json", strip(res)))
    print(json.dumps({"best_cell": best_cell, "G4": {k: v["decision"] for k, v in g4.items()},
                      "G5": {k: v for k, v in g5.items() if not isinstance(v, dict) or "decision" in k}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
