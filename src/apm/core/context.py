"""Bounded context: the latest valid knowledge the APM holds about entities, self state, goals and predictions.

Everything here is a *current* view with explicit freshness. Repeated identical observations replace each other;
they never accumulate (affect dynamics read this view, so an unchanged scene produces an unchanged target).
Capacity limits keep memory bounded regardless of input rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from apm.protocol.messages import (
    SELF_FIELDS,
    EmbodimentProfile,
    EntityObservation,
    GoalState,
    OutcomeHypothesis,
    SelfState,
)

MAX_ENTITIES = 64
MAX_HYPOTHESES = 128
MAX_GOALS = 16


@dataclass
class Track:
    observation: EntityObservation
    first_seen_s: float
    last_seen_s: float


@dataclass
class Context:
    profile: EmbodimentProfile | None = None
    entity_ttl_s: float = 1.5
    self_ttl_s: float = 1.0
    tracks: dict[str, Track] = field(default_factory=dict)
    self_state: SelfState | None = None
    goals: dict[str, GoalState] = field(default_factory=dict)
    hypotheses: dict[str, OutcomeHypothesis] = field(default_factory=dict)

    # -------------------------------------------------- updates
    def observe_entity(self, obs: EntityObservation) -> Track | None:
        """Returns the track evicted to make room (its encounter has ended), if any."""
        t = obs.header.stamp_s
        track = self.tracks.get(obs.track_id)
        evicted = None
        if track is None:
            if len(self.tracks) >= MAX_ENTITIES:
                oldest = min(self.tracks, key=lambda k: self.tracks[k].last_seen_s)
                evicted = self.tracks.pop(oldest)
            self.tracks[obs.track_id] = Track(obs, t, t)
        else:
            track.observation, track.last_seen_s = obs, t
        return evicted

    def observe_self(self, state: SelfState) -> None:
        self.self_state = state

    def observe_goal(self, goal: GoalState) -> None:
        if goal.goal_id not in self.goals and len(self.goals) >= MAX_GOALS:
            raise ValueError("too many goals")
        self.goals[goal.goal_id] = goal

    def observe_hypothesis(self, hypothesis: OutcomeHypothesis) -> None:
        if hypothesis.hypothesis_id not in self.hypotheses and len(self.hypotheses) >= MAX_HYPOTHESES:
            oldest = min(self.hypotheses, key=lambda k: self.hypotheses[k].header.stamp_s)
            del self.hypotheses[oldest]
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis

    def expire(self, now_s: float) -> list[Track]:
        """Drop stale tracks and expired hypotheses; returns the tracks that ended (for encounter memory)."""
        ended = [t for k, t in self.tracks.items() if now_s - t.last_seen_s > self.entity_ttl_s]
        for t in ended:
            del self.tracks[t.observation.track_id]
        for key in [k for k, h in self.hypotheses.items() if now_s - h.header.stamp_s > h.horizon_s]:
            del self.hypotheses[key]
        return ended

    # -------------------------------------------------- views
    def margin(self, name: str, now_s: float) -> float | None:
        """Calibrated margin fraction in [0, 1], or None when unavailable/stale. Never a silent default."""
        state = self.self_state
        if state is None or now_s - state.header.stamp_s > self.self_ttl_s:
            return None
        q = getattr(state, name)
        if not q.usable:
            return None
        if q.unit == "fraction":
            value = float(q.value)
        else:
            calibration = self._calibration(name)
            if calibration is None or calibration.raw_unit != q.unit:
                return None  # a raw value without a matching calibration has no embodiment-independent meaning
            value = calibration.to_margin(float(q.value))
        if name == "actuation_stress":
            value = 1.0 - value  # stress → margin
        return min(1.0, max(0.0, value))

    def _calibration(self, name: str):
        if self.profile is None:
            return None
        return next((c for c in self.profile.calibration if c.field == name), None)

    def body_margins(self, now_s: float) -> dict[str, float]:
        return {name: m for name in SELF_FIELDS if (m := self.margin(name, now_s)) is not None}

    def active_goal_priority(self, goal_id: str | None) -> float | None:
        if goal_id is None:
            return None
        goal = self.goals.get(goal_id)
        return goal.priority if goal is not None and goal.active else None

    def missing_capabilities(self, now_s: float) -> list[str]:
        declared = set(self.profile.self_fields) if self.profile else set(SELF_FIELDS)
        return sorted(name for name in declared if self.margin(name, now_s) is None)
