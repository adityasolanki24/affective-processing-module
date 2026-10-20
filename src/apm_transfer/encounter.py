"""Extended robot-observable encounter record for programme J's passive logger (evaluation and model fitting).

Adds to `apm_revise.evidence.Encounter`:
- the robot's decisions while the person was near: frames within NEAR_M, and how many of them were under a careful
  action (SLOW/YIELD/DETOUR);
- the action context: summed robot speed over observable frames;
- the situation context: bearing at first observable frame (|b| < 30° is frontal), and whether anyone else was within
  NEAR_M of the robot during observable frames (crowding).

Everything is computed from `Obs`; nothing here reads ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from apm_revise.evidence import CAREFUL_ACTIONS, MOVING_MPS, NEAR_M, Encounter

FRONTAL_RAD = math.radians(30.0)
FAST_MPS = 0.35  # robot speed band edge (SLOW mode caps at 0.3 m/s)


@dataclass
class EncounterX(Encounter):
    near_frames: int = 0
    careful_frames: int = 0
    speed_sum: float = 0.0
    bearing0: float | None = None
    crowded_frames: int = 0

    def observe_x(self, distance: float, v: float, action: int, bearing: float, others_near: bool) -> None:
        self.observe(distance, v, action)
        if distance < NEAR_M:
            self.near_frames += 1
            self.careful_frames += action in CAREFUL_ACTIONS
            if abs(v) > MOVING_MPS:
                self.speed_sum += abs(v)
                if self.bearing0 is None:
                    self.bearing0 = bearing
                self.crowded_frames += others_near

    @property
    def careful_fraction(self) -> float | None:
        return self.careful_frames / self.near_frames if self.near_frames else None

    def action_band(self) -> int | None:
        """0 = slow (mean observable-frame speed ≤ FAST_MPS), 1 = fast; None if no observable frame."""
        return None if not self.frames else int(self.speed_sum / self.frames > FAST_MPS)

    def context_band(self) -> int | None:
        """0..3 = (frontal?, crowded?); None if no observable frame."""
        if not self.frames or self.bearing0 is None:
            return None
        frontal = abs(self.bearing0) < FRONTAL_RAD
        crowded = self.crowded_frames > self.frames / 2
        return int(frontal) * 2 + int(crowded)
