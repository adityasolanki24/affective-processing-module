"""Cluster-level statistics for v0.2 (MASTER_PREREGISTRATION.md §2): never treat correlated steps as independent.

* Predictions: per-cluster sums of a per-sample loss, two-level bootstrap (clusters with replacement, then model
  seeds with replacement), metric = ratio of totals, averaged over the resampled seeds.
* Behaviour: per-lifetime totals of paired systems, cluster bootstrap over lifetimes, ratio-of-totals metrics.
Percentile 95% intervals; one-sided bootstrap p-values for Holm adjustment.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 10_000


def cluster_sums(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (cluster ids, per-cluster sum of values, per-cluster count)."""
    ids, inverse = np.unique(groups, return_inverse=True)
    return ids, np.bincount(inverse, weights=values, minlength=len(ids)), np.bincount(inverse, minlength=len(ids))


def _weights(n_boot: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)


def relative_loss_difference(loss_a: np.ndarray, loss_b: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT,
                             seed: int = 0) -> dict:
    """Relative difference of mean loss, (A − B)/B, with a two-level bootstrap.

    loss_a, loss_b: [seeds, samples] per-sample losses of two arms (each row one model seed; rows may differ in
    number between arms); samples aligned across arms; groups: [samples] cluster ids.
    """
    rng = np.random.default_rng(seed)
    loss_a, loss_b = np.atleast_2d(loss_a), np.atleast_2d(loss_b)
    _, sa, n = zip(*[cluster_sums(row, groups) for row in loss_a], strict=True)
    _, sb, _ = zip(*[cluster_sums(row, groups) for row in loss_b], strict=True)
    sa_m, sb_m, counts = np.array(sa), np.array(sb), np.array(n[0], dtype=float)  # [seeds, clusters]
    point = (sa_m.mean(0).sum() / counts.sum()) / (sb_m.mean(0).sum() / counts.sum()) - 1.0
    w = _weights(n_boot, len(counts), rng)
    ka = rng.integers(0, len(sa_m), size=(n_boot, len(sa_m)))
    kb = rng.integers(0, len(sb_m), size=(n_boot, len(sb_m)))
    # per bootstrap: mean over resampled seeds of the per-cluster sums, then weighted totals
    a_tot = np.einsum("bc,bc->b", w, sa_m[ka].mean(1))
    b_tot = np.einsum("bc,bc->b", w, sb_m[kb].mean(1))
    r = a_tot / b_tot - 1.0
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"relative_difference": float(point), "ci95": [float(lo), float(hi)], "se": float(r.std()),
            "p_not_lower": float(np.mean(r >= 0)), "draws": r}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def paired_lifetimes(a: dict[str, np.ndarray], b: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Paired behavioural comparison of systems A and B over the same lifetimes.

    a, b: per-lifetime totals with keys collisions, minutes, goals_completed, goals_issued, idle, periods (arrays
    aligned by lifetime). Returns ratio-of-totals collision-rate change r, completion difference (pp), idle
    difference (pp), each with a paired cluster-bootstrap 95% CI.
    """
    rng = np.random.default_rng(seed)
    n = len(a["collisions"])
    w = np.vstack([np.ones(n), _weights(n_boot, n, rng)])

    def rate(d, num, den):
        return (w @ d[num]) / np.maximum(1e-12, w @ d[den])

    col = rate(a, "collisions", "minutes") / np.maximum(1e-12, rate(b, "collisions", "minutes")) - 1.0
    comp = rate(a, "goals_completed", "goals_issued") - rate(b, "goals_completed", "goals_issued")
    idle = rate(a, "idle", "periods") - rate(b, "idle", "periods")
    out: dict[str, object] = {}
    for name, draws in (("collision_rate_relative", col), ("completion_difference", comp),
                        ("idle_difference", idle)):
        lo, hi = np.percentile(draws[1:], [2.5, 97.5])
        out[name] = {"estimate": float(draws[0]), "ci95": [float(lo), float(hi)], "se": float(draws[1:].std())}
    out["p_collision_not_lower"] = float(np.mean(col[1:] >= 0))
    out["p_completion_not_higher"] = float(np.mean(comp[1:] <= 0))
    return out


def _seed_draws(n_seeds: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """[1 + n_boot, n_seeds] seed indices: row 0 = every seed once (point estimate), then resamples."""
    if n_seeds == 1:
        return np.zeros((n_boot + 1, 1), int)
    return np.vstack([np.arange(n_seeds), rng.integers(0, n_seeds, size=(n_boot, n_seeds))])


def paired_lifetimes_two_level(a: dict[str, np.ndarray], b: dict[str, np.ndarray], n_boot: int = N_BOOT,
                               seed: int = 0) -> dict:
    """`paired_lifetimes` for systems with several training seeds (E4): two-level bootstrap.

    a, b: per-lifetime totals as arrays [seeds, lifetimes] (a deterministic system has one row), lifetimes aligned
    across systems. Each draw resamples lifetimes (shared by both systems: paired), then each system's seeds with
    replacement; per-lifetime totals are averaged over the resampled seeds and combined as ratios of totals.
    """
    rng = np.random.default_rng(seed)
    a = {k: np.atleast_2d(np.asarray(v, float)) for k, v in a.items()}
    b = {k: np.atleast_2d(np.asarray(v, float)) for k, v in b.items()}
    n, sa, sb = a["collisions"].shape[1], a["collisions"].shape[0], b["collisions"].shape[0]
    w = np.vstack([np.ones(n), _weights(n_boot, n, rng)])
    ka = _seed_draws(sa, n_boot, rng)
    kb = _seed_draws(sb, n_boot, rng)

    def tot(d, key, k):  # [draws]: weighted total over lifetimes of the seed-averaged per-lifetime value
        return np.einsum("bn,bn->b", w, d[key][k].mean(1))

    def rate(d, k, num, den):
        return tot(d, num, k) / np.maximum(1e-12, tot(d, den, k))

    col = rate(a, ka, "collisions", "minutes") / np.maximum(1e-12, rate(b, kb, "collisions", "minutes")) - 1.0
    comp = rate(a, ka, "goals_completed", "goals_issued") - rate(b, kb, "goals_completed", "goals_issued")
    idle = rate(a, ka, "idle", "periods") - rate(b, kb, "idle", "periods")
    out: dict[str, object] = {}
    for name, draws in (("collision_rate_relative", col), ("completion_difference", comp),
                        ("idle_difference", idle)):
        lo, hi = np.percentile(draws[1:], [2.5, 97.5])
        out[name] = {"estimate": float(draws[0]), "ci95": [float(lo), float(hi)], "se": float(draws[1:].std())}
    out["p_collision_not_lower"] = float(np.mean(col[1:] >= 0))
    out["p_completion_not_higher"] = float(np.mean(comp[1:] <= 0))
    out["seeds"] = [int(sa), int(sb)]
    return out
