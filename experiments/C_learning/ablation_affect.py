"""EXPLORATORY (post-hoc, not preregistered): which part of the APM's affect inputs carries M5's advantage?

Same data, split, GRU and seed as Milestone C. Extra inputs per sample are [V, A, D, u_threat, u_promise,
u_strain, u_control, u_unknown]. Arms: M4 (canonical features only), PAD only, u only, and PAD + u (= M5).

    python experiments/C_learning/ablation_affect.py --cache runs/nerva_episodes.pkl --out <ablation.json>
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from apm.learning.dataset import samples
from apm.learning.evaluation import bootstrap_difference, summary
from apm.learning.features import WIDTH
from apm.learning.predictor import GRUPredictor

ARMS = {"M4_features": [], "PAD_only": [0, 1, 2], "u_only": [3, 4, 5, 6, 7], "PAD_plus_u_M5": list(range(8))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("runs/nerva_episodes.pkl"))
    ap.add_argument("--out", type=Path, default=Path("experiments/C_learning/ablation.json"))
    args = ap.parse_args()
    episodes = pickle.loads(args.cache.read_bytes())
    ids = np.random.default_rng(0).permutation(len(episodes))
    n = len(ids)
    train = [episodes[i] for i in ids[:int(.6 * n)]]
    val = [episodes[i] for i in ids[int(.6 * n):int(.8 * n)]]
    test = [episodes[i] for i in ids[int(.8 * n):]]
    data = {name: samples(part, "stability_loss", 0.5, 20, use_extra=True) for name, part in
            (("train", train), ("val", val), ("test", test))}
    out, preds = {}, {}
    for arm, cols in ARMS.items():
        keep = list(range(WIDTH)) + [WIDTH + c for c in cols]
        X, y = data["train"][0][..., keep], data["train"][1]
        Xv, yv = data["val"][0][..., keep], data["val"][1]
        Xt, yt, gt = data["test"][0][..., keep], data["test"][1], data["test"][2]
        model = GRUPredictor(len(keep), hidden=64, seed=0).fit(X, y, Xv, yv)
        preds[arm] = model.predict(Xt)
        out[arm] = summary(preds[arm], yt)
        print(arm, round(out[arm]["brier"], 5), round(out[arm]["auroc"], 3), flush=True)
    yt, gt = data["test"][1], data["test"][2]
    out["differences_vs_M4"] = {arm: bootstrap_difference(preds[arm], preds["M4_features"], yt, gt)
                                for arm in ARMS if arm != "M4_features"}
    out["note"] = "exploratory, post-hoc; not a preregistered criterion"
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["differences_vs_M4"], indent=1))


if __name__ == "__main__":
    main()
