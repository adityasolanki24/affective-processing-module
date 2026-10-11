"""Transparent statistical outcome predictors (NumPy only). All share fit(X, y) / predict(X) on [N, history, F]."""
from __future__ import annotations

import numpy as np

from apm.learning.features import NAMES

STABILITY, STABILITY_RATE, GOAL = 0, 1, NAMES.index("goal_moving")
F = len(NAMES)


class GoalRate:
    """M0: outcome memory alone, the event rate per goal type (an analogue of APM's context memory)."""

    def fit(self, X, y):
        goal = X[:, -1, GOAL] * X[:, -1, F + GOAL]
        self.rates = {g: (y[goal == g].sum() + 1) / ((goal == g).sum() + 2) for g in (0.0, 1.0)}
        return self

    def predict(self, X):
        goal = X[:, -1, GOAL] * X[:, -1, F + GOAL]
        return np.where(goal > 0.5, self.rates[1.0], self.rates[0.0])


class MarginBins:
    """M1: empirical frequency by current stability-margin bin × goal moving (Laplace prior)."""

    def __init__(self, bins: int = 10):
        self.bins = bins

    def _key(self, X):
        margin = np.where(X[:, -1, F + STABILITY] > 0, X[:, -1, STABILITY], -1.0)
        b = np.where(margin < 0, self.bins, np.clip((margin * self.bins).astype(int), 0, self.bins - 1))
        return b * 2 + (X[:, -1, GOAL] > 0.5)

    def fit(self, X, y):
        k = self._key(X)
        self.table = np.full(2 * (self.bins + 1), (y.sum() + 1) / (len(y) + 2))
        for key in np.unique(k):
            sel = k == key
            self.table[key] = (y[sel].sum() + 1) / (sel.sum() + 2)
        return self

    def predict(self, X):
        return self.table[self._key(X)]


class Logistic:
    """M2: L2-regularized logistic regression on the current feature vector (Newton/IRLS)."""

    def __init__(self, l2: float = 1e-2, iterations: int = 30):
        self.l2, self.iterations = l2, iterations

    def _design(self, X):
        return np.concatenate([np.ones((len(X), 1)), X[:, -1, :]], axis=1)

    def fit(self, X, y):
        A = self._design(X)
        w = np.zeros(A.shape[1])
        for _ in range(self.iterations):
            p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
            g = A.T @ (p - y) + self.l2 * np.r_[0, w[1:]]
            H = (A * (p * (1 - p))[:, None]).T @ A + self.l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
            w -= np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
        self.w = w
        return self

    def predict(self, X):
        return 1 / (1 + np.exp(-np.clip(self._design(X) @ self.w, -30, 30)))


class TiltExtrapolation:
    """M3: the NERVA adapter's estimator re-expressed on canonical features (biped calibration: 45° ↔ margin 0)."""

    def __init__(self, ahead_s: float = 0.3, threshold_deg: float = 25.0, scale_deg: float = 3.0,
                 full_scale_deg: float = 45.0):
        self.ahead_s, self.threshold, self.scale, self.full = ahead_s, threshold_deg, scale_deg, full_scale_deg

    def fit(self, X, y):
        return self

    def predict(self, X):
        margin, rate = X[:, -1, STABILITY], X[:, -1, STABILITY_RATE]
        valid = X[:, -1, F + STABILITY] > 0
        tilt = self.full * (1 - margin)
        predicted = tilt - self.full * rate * self.ahead_s
        p = 1 / (1 + np.exp(-(predicted - self.threshold) / self.scale))
        return np.where(valid, p, 0.0)


def parameter_count(model) -> int:
    if hasattr(model, "w"):
        return int(model.w.size)
    if hasattr(model, "table"):
        return int(model.table.size)
    if hasattr(model, "rates"):
        return len(model.rates)
    return 0
