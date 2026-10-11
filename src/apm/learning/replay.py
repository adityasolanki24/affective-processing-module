"""Experience replay and the shadow learner (validated model serves; a shadow copy learns; promotion is explicit).

    validated model ──► live predictions
          │
          └──► experience buffer ──► shadow learner ──► frozen evaluation ──► promote or reject
                                                                               │
                                              rollback ◄── last validated model ┘

The live model never changes after a gradient step. A shadow update is promoted only if it improves the frozen
validation score by `min_gain` (relative) without regressing on the protected (older-experience) validation set by
more than `max_regression`. Every promotion keeps the previous model as a rollback target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apm.learning.evaluation import brier


@dataclass
class ReplayBuffer:
    """Bounded store of (X, y) batches tagged by arrival; samples mix recent and older experience."""
    capacity: int = 200_000
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)
    arrival: list = field(default_factory=list)

    def add(self, X, y, arrival: int) -> None:
        self.X.append(X)
        self.y.append(y)
        self.arrival.append(np.full(len(y), arrival))
        while sum(len(v) for v in self.y) > self.capacity and len(self.y) > 1:
            self.X.pop(0), self.y.pop(0), self.arrival.pop(0)

    def mix(self, X_new, y_new, old_fraction: float, rng: np.random.Generator):
        """New batch plus a uniform sample of older experience making up `old_fraction` of the result."""
        if old_fraction <= 0 or not self.y:
            return X_new, y_new
        X_old, y_old = np.concatenate(self.X), np.concatenate(self.y)
        n_old = min(len(y_old), int(round(len(y_new) * old_fraction / (1 - old_fraction))))
        take = rng.choice(len(y_old), n_old, replace=False)
        return np.concatenate([X_new, X_old[take]]), np.concatenate([y_new, y_old[take]])


@dataclass
class ShadowLearner:
    validated: Any                    # model with predict(X), clone(), fit(); never mutated in place
    X_val: np.ndarray
    y_val: np.ndarray
    protected: np.ndarray             # boolean mask over the validation set: older experience to protect
    min_gain: float = 0.02
    max_regression: float = 0.05
    epochs: int = 5
    old_fraction: float = 0.5
    seed: int = 0
    buffer: ReplayBuffer = field(default_factory=ReplayBuffer)
    history: list = field(default_factory=list)
    previous: list = field(default_factory=list)  # rollback stack

    def scores(self, model) -> tuple[float, float]:
        p = model.predict(self.X_val)
        return brier(p, self.y_val), brier(p[self.protected], self.y_val[self.protected])

    def offer(self, X_new, y_new, arrival: int, corrupt: bool = False) -> dict:
        rng = np.random.default_rng(self.seed + arrival)
        y_train = rng.permutation(y_new) if corrupt else y_new
        X_mix, y_mix = self.buffer.mix(X_new, y_train, self.old_fraction, rng)
        shadow = self.validated.clone()
        shadow.fit(X_mix, y_mix, epochs=self.epochs, early_stopping=False)
        before, before_protected = self.scores(self.validated)
        after, after_protected = self.scores(shadow)
        promote = (after <= before * (1 - self.min_gain)
                   and after_protected <= before_protected * (1 + self.max_regression))
        record = {"arrival": arrival, "corrupted": corrupt, "samples": int(len(y_mix)),
                  "val_brier_before": before, "val_brier_after": after,
                  "protected_before": before_protected, "protected_after": after_protected, "promoted": bool(promote)}
        if promote:
            self.previous.append(self.validated)
            self.validated = shadow
        if not corrupt:
            self.buffer.add(X_new, y_new, arrival)  # experience is kept even when the shadow is rejected
        self.history.append(record)
        return record

    def rollback(self) -> bool:
        if not self.previous:
            return False
        self.validated = self.previous.pop()
        self.history.append({"rollback": True})
        return True
