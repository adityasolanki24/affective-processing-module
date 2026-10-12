"""Independent safety monitor for expressive motion (programme H §1).

Inputs are physical state only (time, tilt, angular speed, joint positions and limits, actuator forces and limits):
it never sees intents, memory or internal state, so nothing expressive can delay or suppress it (the signature
enforces this; tests/test_h_express.py checks it).

Trip rule = NERVA's deterministic supervisor (`nerva/safety.py`: stop at tilt >= 25 deg or stability risk >= 0.75,
i.e. angular speed >= 5 rad/s; resume after tilt <= 12 deg held for 0.5 s), plus two diagnostics that become trip
conditions only when an addendum fixes their thresholds from pilot data (`joint_margin_rad`, `saturation_steps`).
While tripped the controller must command neutral rest with zero expressive offsets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STOP_TILT_DEG = 25.0
STOP_ANGULAR_SPEED = 5.0  # rad/s: NERVA stability_risk = (w - 2)/4 >= 0.75
RESUME_TILT_DEG = 12.0
RESUME_HOLD_S = 0.5


@dataclass(frozen=True)
class MonitorLimits:
    stop_tilt_deg: float = STOP_TILT_DEG
    stop_angular_speed: float = STOP_ANGULAR_SPEED
    resume_tilt_deg: float = RESUME_TILT_DEG
    resume_hold_s: float = RESUME_HOLD_S
    joint_margin_rad: float | None = None  # trip if any actuated joint is closer than this to a range limit
    saturation_steps: int | None = None  # trip if any actuator is at its force limit this many consecutive steps
    saturation_tol: float = 0.98  # |force| >= tol * limit counts as saturated


@dataclass
class SafetyMonitor:
    limits: MonitorLimits = field(default_factory=MonitorLimits)
    tripped: bool = False
    trips: int = 0
    reasons: list[tuple[float, str]] = field(default_factory=list)
    _calm_since: float | None = None
    _saturated_run: np.ndarray | None = None
    min_joint_margin: float = float("inf")
    max_saturated_run: int = 0

    def update(self, t: float, tilt_deg: float, angular_speed: float, joint_pos=None, joint_lo=None,
               joint_hi=None, actuator_force=None, force_limit=None) -> bool:
        """Advance one control step; returns True while the fallback must be active."""
        lim = self.limits
        reason = ""
        if not (np.isfinite(tilt_deg) and np.isfinite(angular_speed)):
            reason = "non-finite state"
        elif tilt_deg >= lim.stop_tilt_deg:
            reason = f"tilt {tilt_deg:.1f} deg"
        elif angular_speed >= lim.stop_angular_speed:
            reason = f"angular speed {angular_speed:.2f} rad/s"
        if joint_pos is not None and joint_lo is not None and joint_hi is not None:
            q = np.asarray(joint_pos, dtype=float)
            margin = float(np.min(np.minimum(q - np.asarray(joint_lo), np.asarray(joint_hi) - q)))
            self.min_joint_margin = min(self.min_joint_margin, margin)
            if not reason and lim.joint_margin_rad is not None and margin < lim.joint_margin_rad:
                reason = f"joint margin {margin:.3f} rad"
        if actuator_force is not None and force_limit is not None:
            sat = np.abs(np.asarray(actuator_force, dtype=float)) >= lim.saturation_tol * np.asarray(force_limit)
            if self._saturated_run is None:
                self._saturated_run = np.zeros(sat.shape, dtype=int)
            self._saturated_run = np.where(sat, self._saturated_run + 1, 0)
            run = int(self._saturated_run.max())
            self.max_saturated_run = max(self.max_saturated_run, run)
            if not reason and lim.saturation_steps is not None and run >= lim.saturation_steps:
                reason = f"force saturation {run} steps"
        if reason:
            if not self.tripped:
                self.trips += 1
                self.reasons.append((float(t), reason))
            self.tripped, self._calm_since = True, None
        elif self.tripped:
            if tilt_deg <= lim.resume_tilt_deg:
                self._calm_since = t if self._calm_since is None else self._calm_since
                if t - self._calm_since >= lim.resume_hold_s - 1e-9:
                    self.tripped, self._calm_since = False, None
            else:
                self._calm_since = None
        return self.tripped
