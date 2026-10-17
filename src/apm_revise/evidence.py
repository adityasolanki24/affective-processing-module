"""Robot-observable encounter evidence for programme I.

An *encounter* is a person track's lifetime (ends when unseen for more than ENTITY_TTL_S, as in F and G). Everything
here is computed from `Obs` only; no ground truth is used.

During an encounter the robot records:
- whether a collision with the track occurred;
- the number of *observable* frames: the robot moves faster than MOVING_MPS while the person is within NEAR_M. In the
  wheeled testbed people react to a moving robot only, so only these frames show how a person behaves;
- the minimum clearance over observable frames;
- whether the robot chose a careful action (SLOW/YIELD/DETOUR) while the person was within NEAR_M (G's "avoided").

Outcome coding:
- **dense:** 8 categories: collision, or the minimum observable clearance in one of the bins
  [0, .05), [.05, .10), [.10, .15), [.15, .20), [.20, .30), [.30, .50), [.50, ∞) m;
- **sparse:** 2 categories: collision or not (G's collision-only evidence, with the same observability gate).

An encounter with fewer than MIN_FRAMES observable frames and no collision is *censored*: it carries no evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from apm_testbeds.wheeled_ctrl import DETOUR, SLOW, YIELD

ENTITY_TTL_S = 1.5
NEAR_M, MOVING_MPS, MIN_FRAMES, RELIABLE_CONF = 1.5, 0.15, 3, 0.5
EXPOSED_M = 0.5  # G's exposure radius, kept only to report G's encounter classes alongside
GAP_EDGES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
CODINGS = {"dense": 1 + len(GAP_EDGES) + 1, "sparse": 2}
CAREFUL_ACTIONS = (SLOW, YIELD, DETOUR)


@dataclass
class Encounter:
    track: str
    start: float
    identity: str | None = None
    confidence: float = 0.0
    frames: int = 0
    min_gap: float = float("inf")
    collision: bool = False
    exposed: bool = False   # G's definition: within EXPOSED_M while moving
    avoided: bool = False   # G's definition: careful action while within NEAR_M
    p_start: float | None = None

    def observe(self, distance: float, v: float, action: int) -> None:
        if distance < NEAR_M:
            if abs(v) > MOVING_MPS:
                self.frames += 1
                self.min_gap = min(self.min_gap, distance)
            if action in CAREFUL_ACTIONS:
                self.avoided = True
        if distance < EXPOSED_M and abs(v) > MOVING_MPS:
            self.exposed = True

    @property
    def reliable(self) -> bool:
        return self.identity is not None and self.confidence >= RELIABLE_CONF

    def g_class(self) -> str:
        """G's classification (collision / exposed / avoided / unexposed), for comparison with G's reports."""
        return "collision" if self.collision else "exposed" if self.exposed else \
            "avoided" if self.avoided else "unexposed"

    def category(self, coding: str) -> int | None:
        """Outcome category under `coding`, or None if the encounter is censored."""
        if self.collision:
            return 0
        if self.frames < MIN_FRAMES:
            return None
        if coding == "sparse":
            return 1
        return 1 + sum(self.min_gap >= e for e in GAP_EDGES)
