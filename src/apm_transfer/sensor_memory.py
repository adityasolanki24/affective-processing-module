"""Small revisable memory of observable motion events, separate from risk estimation.

The old type model's perturbation readout is identically zero on NERVA. This
companion records a simpler, directly identifiable expectation instead: how often
the camera sees a substantial movement toward the robot during a sensing window.
It never consumes collision truth, entity labels or an intended expression name.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SensorExpectation:
    # Pilot threshold is 5 cm motion threshold plus 3 x 3 cm navigation uncertainty.
    # This changes the estimand to substantial observed movement, not contact risk.
    minimum_reach_m: float = 0.14
    retention: float = 0.9
    counts: dict[str, list[float]] = field(default_factory=dict)

    def observe(self, identity: str, reaches: list[list[float]] | None) -> None:
        if reaches is None or not reaches:
            return  # no observed windows is not evidence of no motion
        alpha, beta = self.counts.get(identity, [1., 1.])
        alpha, beta = 1 + self.retention * (alpha - 1), 1 + self.retention * (beta - 1)
        for window in reaches:
            values = np.asarray(window, float)
            event = bool(np.any(np.isfinite(values) & (values >= self.minimum_reach_m)))
            alpha += int(event)
            beta += int(not event)
        self.counts[identity] = [alpha, beta]

    def readout(self, identity: str) -> dict:
        a, b = self.counts.get(identity, [1., 1.])
        return {"movement_probability": a / (a + b), "effective_windows": a + b - 2,
                "uncertainty": float(np.sqrt(12 * a * b / ((a + b) ** 2 * (a + b + 1))))}

    def state(self) -> dict:
        return {"minimum_reach_m": self.minimum_reach_m, "retention": self.retention, "counts": self.counts}

    def load(self, state: dict) -> None:
        self.minimum_reach_m = float(state["minimum_reach_m"])
        self.retention = float(state["retention"])
        self.counts = {k: [float(x) for x in v] for k, v in state["counts"].items()}
