"""Shared builders for synthetic AffectLink traffic (clearly synthetic; no robot data)."""
from __future__ import annotations

import pytest

from apm import APM
from apm.adapters.base import Emitter
from apm.protocol.messages import (
    EmbodimentProfile,
    EntityObservation,
    GoalState,
    MarginCalibration,
    OutcomeHypothesis,
    OutcomeSignal,
    Quantity,
    SelfState,
)

KINDS = ("collision", "stability_loss", "task_completed", "task_interrupted", "benign_contact", "recovery")


class Robot:
    """A synthetic robot-side sender."""

    def __init__(self, source: str = "testbot"):
        self.emit = Emitter(source)
        self.profile = EmbodimentProfile(
            header=self.emit.header("profile", 0.0), robot_id=source, embodiment_class="synthetic",
            self_fields=("stability_margin", "energy_margin"), outcome_kinds=KINDS, goal_types=("reach",),
            calibration=(MarginCalibration(field="energy_margin", raw_unit="V", nominal=12.6, critical=10.5,
                                           meaning="battery voltage"),))

    def apm(self, **kwargs) -> APM:
        return APM(profile=self.profile, **kwargs)

    def self_state(self, t, stability=0.9, energy_v=12.4, mode="walk"):
        return SelfState(header=self.emit.header("self_state", t),
                         stability_margin=Quantity.missing("fraction") if stability is None
                         else Quantity.measured(stability, "fraction"),
                         energy_margin=Quantity.missing("V") if energy_v is None else Quantity.measured(energy_v, "V"),
                         mode=mode)

    def goal(self, t, priority=0.8, active=True):
        return GoalState(header=self.emit.header("goal", t), goal_id="g1", goal_type="reach", priority=priority,
                         active=active, progress=Quantity.measured(0.3, "fraction"))

    def entity(self, t, track="t1", identity=None, confidence=0.0, distance=1.0, closing=0.0, entity_type="person"):
        return EntityObservation(header=self.emit.header("entities", t), track_id=track, identity=identity,
                                 identity_confidence=confidence, entity_type=entity_type,
                                 distance=Quantity.measured(distance, "m"),
                                 closing_speed=Quantity.measured(closing, "m/s"))

    def hypothesis(self, t, kind="collision", target="t1", p=0.7, horizon=2.0, hid="h1", affected="self"):
        return OutcomeHypothesis(header=self.emit.header("hypotheses", t), hypothesis_id=hid, kind=kind,
                                 target=target, affected=affected, probability=p, horizon_s=horizon)

    def outcome(self, t, kind="collision", target="t1", magnitude=0.8, oid="o1", confirmed=True, affected="self"):
        return OutcomeSignal(header=self.emit.header("outcomes", t), outcome_id=oid, kind=kind, target=target,
                             affected=affected, magnitude=magnitude, confirmed=confirmed)


@pytest.fixture
def robot() -> Robot:
    return Robot()
