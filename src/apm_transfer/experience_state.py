"""Per-person experience memory from the robot's own sensing, and its read-out as an experience state.

Two learned memories, both revisable Beta-Bernoulli expectations with exponential retention:

- **Disturbance** (proprioceptive): was the robot moved while it was commanded to stand still? Per sensing
  window, the robot measures its base's peak horizontal speed while its locomotion command is zero (odometry). A
  window is a disturbance event if that peak exceeds the robot's own standing-sway threshold. The threshold is
  calibrated, without labels, from the robot's standing samples with nobody within reach (the settle phase before
  each approach).
- **Movement toward the robot** (camera): `SensorExpectation`, substantial approach movements seen by the depth
  eye.

Neither memory sees simulator truth: no contact reports, person types or intended expression names.

`ExperienceState.readout` combines them into the quantities the expressive program uses:
- wariness, the expected chance of being disturbed or approached forcefully;
- uncertainty, from the evidence count;
- familiarity, from the effective number of meetings.

The combination weights are designed, not learned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apm_transfer.sensor_memory import SensorExpectation


@dataclass
class DisturbanceMemory:
    threshold_m_s: float = 0.05  # per-robot; replaced by the label-free calibration (calibrate_threshold)
    retention: float = 0.85  # per meeting: old evidence fades, so outdated tendencies can be revised
    counts: dict[str, list[float]] = field(default_factory=dict)
    meetings: dict[str, float] = field(default_factory=dict)

    def observe(self, identity: str, window_peaks: list[float] | None) -> None:
        """One meeting: the peak unintended base speed of each sensing window (None: no valid window)."""
        if not window_peaks:
            return
        a, b = self.counts.get(identity, [1.0, 1.0])
        a, b = 1 + self.retention * (a - 1), 1 + self.retention * (b - 1)
        disturbed = any(np.isfinite(p) and p > self.threshold_m_s for p in window_peaks)
        # one event per meeting: a single push and repeated pushes are both "this meeting disturbed me"
        a += float(disturbed)
        b += float(not disturbed)
        self.counts[identity] = [a, b]
        self.meetings[identity] = self.retention * self.meetings.get(identity, 0.0) + 1.0

    def probability(self, identity: str) -> float:
        a, b = self.counts.get(identity, [1.0, 1.0])
        return a / (a + b)

    def state(self) -> dict:
        return {"threshold_m_s": self.threshold_m_s, "retention": self.retention, "counts": self.counts,
                "meetings": self.meetings}

    def load(self, s: dict) -> None:
        self.threshold_m_s, self.retention = float(s["threshold_m_s"]), float(s["retention"])
        self.counts = {k: [float(x) for x in v] for k, v in s["counts"].items()}
        self.meetings = {k: float(v) for k, v in s.get("meetings", {}).items()}


def calibrate_threshold(standing_peaks: list[float], quantile: float = 0.99, margin: float = 1.5) -> float:
    """Label-free threshold: a margin above the robot's own standing-sway peak speed with nobody within reach."""
    x = np.asarray([p for p in standing_peaks if np.isfinite(p)], float)
    if len(x) < 10:
        raise ValueError("need at least 10 standing samples to calibrate")
    return float(margin * np.quantile(x, quantile))


@dataclass
class ExperienceState:
    disturbance: DisturbanceMemory = field(default_factory=DisturbanceMemory)
    movement: SensorExpectation = field(default_factory=SensorExpectation)
    w_disturbance: float = 0.75  # designed weights of the two learned expectations in "wariness"
    w_movement: float = 0.25

    def observe(self, identity: str, window_peaks: list[float] | None, camera_reaches) -> None:
        self.disturbance.observe(identity, window_peaks)
        self.movement.observe(identity, camera_reaches)

    def readout(self, identity: str) -> dict:
        d = self.disturbance.probability(identity)
        m = self.movement.readout(identity)["movement_probability"]
        a, b = self.disturbance.counts.get(identity, [1.0, 1.0])
        sd = float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
        n = self.disturbance.meetings.get(identity, 0.0)
        return {"wariness": float(self.w_disturbance * d + self.w_movement * m),
                "uncertainty": float(min(1.0, sd / np.sqrt(1 / 12))),  # 1 = the uniform prior's spread
                "familiarity": float(1.0 - np.exp(-n / 3.0)), "disturbance_p": d, "movement_p": m,
                "meetings": n}

    def state(self) -> dict:
        return {"disturbance": self.disturbance.state(), "movement": self.movement.state(),
                "weights": [self.w_disturbance, self.w_movement]}

    def load(self, s: dict) -> None:
        self.disturbance.load(s["disturbance"])
        self.movement.load(s["movement"])
        self.w_disturbance, self.w_movement = (float(x) for x in s["weights"])
