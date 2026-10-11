"""Canonical learning features computed from AffectLink messages (embodiment-independent by construction).

Each feature has a validity mask; a missing measurement contributes value 0 with mask 0, never a fake value.
  0 stability_margin       calibrated fraction
  1 stability_rate         d/dt of the above, per second, clipped to ±5
  2 actuation_margin       1 − actuation stress
  3 actuation_rate         d/dt, clipped to ±5
  4 energy_margin          calibrated fraction
  5 entity_proximity       exp(−nearest distance / 1 m) over currently tracked entities
  6 entity_closing         closing speed of the nearest entity, m/s / 2, clipped to ±1
  7 goal_moving            1 if the most important active goal is a motion goal, else 0
"""
from __future__ import annotations

import math

import numpy as np

from apm.core.context import Context
from apm.protocol.messages import EmbodimentProfile, EntityObservation, GoalState, SelfState

NAMES = ("stability_margin", "stability_rate", "actuation_margin", "actuation_rate", "energy_margin",
         "entity_proximity", "entity_closing", "goal_moving")
WIDTH = 2 * len(NAMES)  # values then masks
MOTION_GOALS = ("track_velocity", "navigate", "reach", "follow")


class FeatureTracker:
    def __init__(self, profile: EmbodimentProfile):
        self.context = Context(profile=profile)
        self._prev: dict[str, tuple[float, float]] = {}

    def ingest(self, message) -> None:
        if isinstance(message, SelfState):
            self.context.observe_self(message)
        elif isinstance(message, GoalState):
            self.context.observe_goal(message)
        elif isinstance(message, EntityObservation):
            self.context.observe_entity(message)

    def vector(self, now_s: float) -> np.ndarray:
        self.context.expire(now_s)
        values = np.zeros(len(NAMES))
        masks = np.zeros(len(NAMES))
        for i, name, rate_i in ((0, "stability_margin", 1), (2, "actuation_stress", 3)):
            m = self.context.margin(name, now_s)
            if m is None:
                self._prev.pop(name, None)
                continue
            values[i], masks[i] = m, 1.0
            prev = self._prev.get(name)
            if prev is not None and now_s > prev[0]:
                values[rate_i] = float(np.clip((m - prev[1]) / (now_s - prev[0]), -5, 5))
                masks[rate_i] = 1.0
            self._prev[name] = (now_s, m)
        energy = self.context.margin("energy_margin", now_s)
        if energy is not None:
            values[4], masks[4] = energy, 1.0
        nearest = None
        for track in self.context.tracks.values():
            d = track.observation.distance
            if d.usable and d.value is not None and (nearest is None or d.value < nearest[0]):
                nearest = (float(d.value), track.observation.closing_speed)
        if nearest is not None:
            values[5], masks[5] = math.exp(-nearest[0]), 1.0
            closing = nearest[1].value
            if nearest[1].usable and closing is not None:
                values[6], masks[6] = float(np.clip(closing / 2.0, -1, 1)), 1.0
        goals = sorted((g for g in self.context.goals.values() if g.active), key=lambda g: -g.priority)
        if goals:
            values[7], masks[7] = float(goals[0].goal_type in MOTION_GOALS), 1.0
        return np.concatenate([values, masks])
