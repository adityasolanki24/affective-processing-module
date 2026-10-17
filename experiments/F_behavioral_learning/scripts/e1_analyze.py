"""E1 analysis: metrics per arm, baseline selection on V, primary contrasts (Holm), secondary analyses, power.

    python scripts/e1_analyze.py --preds runs/E1/preds --embodiment wheeled --target collision --part P [--power]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from e1_train import ARMS, NERVA_SKIP  # noqa: E402

from apm.learning.evaluation import auroc, ece, log_loss, recall_at_fpr  # noqa: E402
from apm_research.provenance import write_json  # noqa: E402
from apm_research.stats import holm, relative_loss_difference  # noqa: E402

PRIMARY = ("U", "P", "UP")
MARGIN = 0.05


def load(preds: Path, emb: str, target: str, part: str):
    arms = {}
    for arm in ARMS:
        if emb == "nerva" and arm in NERVA_SKIP:
            continue
        files = sorted(preds.glob(f"{emb}_{target}_{arm}_s*.npz"))
        if not files:
            continue
        runs = [np.load(f) for f in files]
        if f"{part}_p" not in runs[0]:
            continue
        arms[arm] = {"p": np.stack([r[f"{part}_p"] for r in runs]), "y": runs[0][f"{part}_y"],
                     "g": runs[0][f"{part}_g"], "benign": runs[0][f"{part}_benign"],
                     "val": [float(np.mean((r["V_p"] - r["V_y"]) ** 2)) for r in runs],
                     "meta": [json.loads(f.with_suffix(".json").read_text()) for f in files]}
    return arms


def metrics(p: np.ndarray, y: np.ndarray, benign: np.ndarray) -> dict:
    rows = []
    for row in p:
        neg_benign = benign & (y == 0)
        rows.append({"brier": float(np.mean((row - y) ** 2)), "log_loss": log_loss(row, y), "ece": ece(row, y),
                     "auroc": auroc(row, y), "recall_at_5pct_fpr": recall_at_fpr(row, y),
                     "benign_false_alarm_rate": float(np.mean(row[neg_benign] >= 0.5)) if neg_benign.any() else None})
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if r[k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
        out[k + "_seed_sd"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    return out


def classify(c: dict) -> str:
    lo, hi = c["ci95_holm"]
    if c["relative_difference"] <= -MARGIN and hi < 0 and c.get("beats_N_and_C", False):
        return "adds meaningful information"
    if lo > -MARGIN and hi < MARGIN:
        return "no meaningful information (CI within ±5%)"
    return "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=Path("runs/E1/preds"))
    ap.add_argument("--embodiment", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--part", default="X")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--power", action="store_true", help="pilot: report bootstrap SEs and required clusters only")
    args = ap.parse_args()
    arms = load(args.preds, args.embodiment, args.target, args.part)
    y, g, benign = arms["R"]["y"], arms["R"]["g"], arms["R"]["benign"]
    for a in arms.values():
        assert np.array_equal(a["y"], y) and np.array_equal(a["g"], g)
    candidates = [a for a in (("R", "RL", "M", "RLM") if args.embodiment == "wheeled" else ("R", "RL")) if a in arms]
    baseline = min(candidates, key=lambda a: np.mean(arms[a]["val"]))
    sq = {a: (v["p"] - y) ** 2 for a, v in arms.items()}
    contrasts, pvals = {}, {}
    for arm in PRIMARY:
        c = relative_loss_difference(sq[arm], sq[baseline], g)
        draws = c.pop("draws")
        c["vs_N"] = relative_loss_difference(sq[arm], sq["N"], g)["relative_difference"]
        c["vs_C"] = relative_loss_difference(sq[arm], sq["C"], g)["relative_difference"]
        c["beats_N_and_C"] = bool(c["vs_N"] <= -MARGIN and c["vs_C"] <= -MARGIN)
        c["_draws"] = draws
        contrasts[arm], pvals[arm] = c, c["p_not_lower"]
    adjusted = holm(pvals)
    order = sorted(pvals, key=lambda k: pvals[k])
    for i, arm in enumerate(order):  # Holm-stepped interval: the k-th smallest p uses alpha/(m − k)
        alpha = 0.05 / (len(order) - i)
        draws = contrasts[arm].pop("_draws")
        contrasts[arm]["p_holm"] = adjusted[arm]
        contrasts[arm]["ci95_holm"] = [float(np.percentile(draws, 100 * alpha / 2)),
                                       float(np.percentile(draws, 100 * (1 - alpha / 2)))]
    if args.power:
        n = len(np.unique(g))
        se = max(c["se"] for c in contrasts.values())
        need = int(np.ceil(n * (2.8 * se / MARGIN) ** 2))
        print(json.dumps({"clusters": n, "max_se": se, "required_clusters_80pct_power": need,
                          "se": {k: c["se"] for k, c in contrasts.items()}, "baseline": baseline}, indent=1))
        return
    for c in contrasts.values():
        c["decision"] = classify(c)
    secondary = {}
    for a, b in (("M", "R"), ("R", "G"), ("RL", "R"), ("RLM", "R"), ("N", "R"), ("C", "R"), ("LR", "R")):
        if a in sq and b in sq:
            d = relative_loss_difference(sq[a], sq[b], g)
            d.pop("draws")
            secondary[f"{a}_vs_{b}"] = d
    result = {"embodiment": args.embodiment, "target": args.target, "partition": args.part,
              "clusters": int(len(np.unique(g))), "samples": int(len(y)), "positives": int(y.sum()),
              "baseline_selected_on_V": baseline,
              "validation_brier": {a: float(np.mean(v["val"])) for a, v in arms.items()},
              "metrics": {a: metrics(v["p"], y, benign) for a, v in arms.items()},
              "params": {a: v["meta"][0]["params"] for a, v in arms.items()},
              "infer_us_per_sample": {a: float(np.mean([m.get(f"{args.part}_infer_us_per_sample", np.nan)
                                                        for m in v["meta"]])) for a, v in arms.items()},
              "primary": contrasts, "secondary_unadjusted": secondary}
    out = args.out or HERE / "results" / "E1" / f"{args.embodiment}_{args.target}_{args.part}.json"
    print(write_json(out, result))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("relative_difference", "ci95_holm", "decision",
                                                                      "vs_N", "vs_C")}
                      for k, v in contrasts.items()}, indent=1))


if __name__ == "__main__":
    main()
