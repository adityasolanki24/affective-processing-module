"""Latent person types learned from the robot's own unlabelled experience, and change-aware beliefs over them.

**Type model** (`TypeModel`, fitted by `fit_types`). Each person (one identity within one lifetime) belongs to one of
K latent types. A type k has:
- a prior share π_k;
- a categorical distribution E_k over encounter outcome categories (`evidence.CODINGS`);
- a per-encounter collision rate r_k over *all* encounters, including censored ones. r_k is on the same scale as
  F's cumulative memory, so the decision rule's memory input keeps its meaning.

The model is fitted by EM with restarts on outcome counts per person. No dispositions or other ground truth are
used. K can be chosen by BIC.

**Belief** (`TypeBelief`). A posterior over types per identity, under a continuous-time Markov change model. A
person may change type at rate λ, re-drawing from π. Between encounters the belief therefore relaxes toward π:
b(t + Δ) = π + (b(t) − π)·e^(−λΔ). This is the exact prediction step of that chain, and it is the evidence-free
forgetting G2b found necessary. At an informative encounter: b ∝ b ⊙ E[:, c]^w, where w is identity confidence.
λ = 0 gives a static mixture (no forgetting).

Outputs: p_collision = b·r; confidence = 1 − H(b)/log K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SMOOTH = 0.5  # Dirichlet pseudo-count per category in the M-step


@dataclass
class TypeModel:
    coding: str
    pi: np.ndarray        # [K]
    E: np.ndarray         # [K, C] outcome-category probabilities per type
    r: np.ndarray         # [K] collision rate per encounter (all encounters)
    loglik: float = float("nan")
    n_units: int = 0

    @property
    def K(self) -> int:
        return len(self.pi)

    def bic(self) -> float:
        C = self.E.shape[1]
        n_par = (self.K - 1) + self.K * (C - 1)
        return -2.0 * self.loglik + n_par * math.log(max(1, self.n_units))

    def to_dict(self) -> dict:
        return {"coding": self.coding, "pi": self.pi.tolist(), "E": self.E.tolist(), "r": self.r.tolist(),
                "loglik": self.loglik, "n_units": self.n_units}

    @classmethod
    def from_dict(cls, d: dict) -> TypeModel:
        return cls(d["coding"], np.asarray(d["pi"], float), np.asarray(d["E"], float), np.asarray(d["r"], float),
                   d.get("loglik", float("nan")), d.get("n_units", 0))


def fit_types(counts: np.ndarray, encounters: np.ndarray, collisions: np.ndarray, K: int, coding: str,
              restarts: int = 10, iters: int = 300, seed: int = 0) -> TypeModel:
    """EM for a K-type latent class model.

    counts[i, c]: (confidence-weighted) informative encounters of person i in category c.
    encounters[i], collisions[i]: all encounters and collisions of person i (for r).
    Types are ordered by collision rate, so type 0 is the safest.
    """
    counts = np.asarray(counts, float)
    n, C = counts.shape
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for _ in range(restarts):
        pi = np.full(K, 1.0 / K)
        E = rng.dirichlet(np.ones(C), size=K)
        ll = -np.inf
        for _ in range(iters):
            L = counts @ np.log(E).T + np.log(pi)            # [n, K]
            m = L.max(1, keepdims=True)
            post = np.exp(L - m)
            z = post.sum(1, keepdims=True)
            gamma = post / z
            new_ll = float((m[:, 0] + np.log(z[:, 0])).sum())
            pi = np.clip(gamma.mean(0), 1e-6, None)
            pi /= pi.sum()
            E = gamma.T @ counts + SMOOTH
            E /= E.sum(1, keepdims=True)
            if abs(new_ll - ll) < 1e-8:
                ll = new_ll
                break
            ll = new_ll
        if best is None or ll > best[0]:
            best = (ll, pi, E, gamma)
    assert best is not None
    ll, pi, E, gamma = best
    r = (gamma.T @ np.asarray(collisions, float)) / np.maximum(1e-9, gamma.T @ np.asarray(encounters, float))
    order = np.argsort(r)
    return TypeModel(coding, pi[order], E[order], r[order], ll, n)


class TypeBelief:
    """Posterior over latent types for one identity, with evidence-free relaxation toward π at rate lam (1/s)."""

    def __init__(self, model: TypeModel, lam: float, t: float):
        self.model, self.lam = model, lam
        self.b = model.pi.copy()
        self.t = t
        self.n = 0.0

    def at(self, t: float) -> np.ndarray:
        pi = self.model.pi
        return pi + (self.b - pi) * math.exp(-self.lam * max(0.0, t - self.t))

    def update(self, category: int, w: float, t: float) -> None:
        b = self.at(t) * self.model.E[:, category] ** w
        self.b = b / b.sum()
        self.t = t
        self.n += w

    def p(self, t: float) -> tuple[float, float]:
        return readout(self.model, self.at(t))


def readout(model: TypeModel, b: np.ndarray) -> tuple[float, float]:
    """(collision probability per encounter, confidence) for a type posterior b."""
    p = float(b @ model.r)
    if model.K < 2:
        return p, 0.0
    h = -float(np.sum(b * np.log(np.clip(b, 1e-12, 1.0))))
    return p, 1.0 - h / math.log(model.K)
