"""Predictive evaluation: proper scores, calibration, discrimination, rare-event detection, grouped bootstrap."""
from __future__ import annotations

import numpy as np


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def auroc(p, y) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
    pos, neg = p[y], p[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    values = np.concatenate([pos, neg])
    # average ranks for ties (vectorised; identical to averaging the ordinal ranks of each tied group)
    uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])  # 0-based start of each tied group in sorted order
    ranks = (first + (counts + 1) / 2.0)[inverse]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_fpr(p, y, fpr: float = 0.05) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
    if not y.any() or y.all():
        return float("nan")
    threshold = np.quantile(p[~y], 1 - fpr)
    return float(np.mean(p[y] > threshold))


def summary(p, y) -> dict:
    return {"brier": brier(p, y), "log_loss": log_loss(p, y), "ece": ece(p, y), "auroc": auroc(p, y),
            "recall_at_5pct_fpr": recall_at_fpr(p, y), "n": int(len(y)), "positives": int(np.sum(y))}


def bootstrap_difference(pa, pb, y, groups, metric=brier, n: int = 1000, seed: int = 0) -> dict:
    """metric(a) − metric(b) with a trial-level (grouped) bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    pa, pb, y, groups = (np.asarray(v) for v in (pa, pb, y, groups))
    ids = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in ids}
    diffs = []
    for _ in range(n):
        take = np.concatenate([index[g] for g in rng.choice(ids, len(ids), replace=True)])
        diffs.append(metric(pa[take], y[take]) - metric(pb[take], y[take]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": metric(pa, y) - metric(pb, y), "ci95": [float(lo), float(hi)]}
