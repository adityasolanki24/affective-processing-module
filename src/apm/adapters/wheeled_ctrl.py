"""Adapter for the v0.2 intervention-capable wheeled testbed (`apm_testbeds.wheeled_ctrl`) → AffectLink v1.

Same capabilities, outcome kinds and TTC estimator formula as the v0.1 wheeled adapter (`apm.adapters.wheeled`), with
two audited differences (experiments/F_behavioral_learning/AUDIT.md):
  D3  a collision hypothesis is *retracted* (probability 0, short horizon) as soon as its entity stops closing, so the
      APM never appraises a stale prediction;
  D4  collisions arrive as contact episodes (the testbed's 1.0 s per-entity refractory), not contact steps.
Severe collisions (robot speed ≥ 0.2 m/s at contact) have magnitude 1.0, others 0.8. Goal timeouts are
`task_interrupted`. The adapter reads `Obs` only, never simulator ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from apm.adapters.base import Emitter
from apm.adapters.wheeled import OUTCOMES, V_MAX, profile
from apm.protocol.messages import (
    EmbodimentProfile,
    EntityObservation,
    GoalState,
    OutcomeHypothesis,
    OutcomeSignal,
    Provenance,
    Quantity,
    SelfState,
    Status,
)

__all__ = ["OUTCOMES", "WheeledCtrlAdapter"]


@dataclass
class WheeledCtrlAdapter:
    emitter: Emitter
    counter: int = 0
    _goal_index: int | None = None
    _goal_sent_s: float = -1e9
    _live: set = field(default_factory=set)   # tracks with a live (non-zero) collision hypothesis
    _in_slip: bool = False
    _calm_since: float | None = None
    _battery_warned: bool = False

    def profile(self) -> EmbodimentProfile:
        return profile(self.emitter)

    def _outcome(self, t: float, kind: str, magnitude: float, target=None, affected="self") -> OutcomeSignal:
        self.counter += 1
        return OutcomeSignal(header=self.emitter.header("outcomes", t),
                             outcome_id=f"{self.emitter.source}:{self.counter}", kind=kind,
                             magnitude=magnitude, target=target, affected=affected, evidence_quality=1.0,
                             confirmed=True, provenance=Provenance(origin="wheeled-ctrl-adapter/detector"))

    def step(self, obs, t: float) -> list:
        """`obs`: an `apm_testbeds.wheeled_ctrl.Obs`; `t`: the absolute (lifetime) time stamp to use."""
        out: list = [SelfState(
            header=self.emitter.header("self_state", t),
            stability_margin=Quantity.measured(obs.traction_margin, "fraction"),
            energy_margin=Quantity.measured(obs.energy, "fraction"),
            actuation_stress=Quantity.measured(min(1.0, abs(obs.v) / V_MAX), "fraction"),
            thermal_margin=Quantity.missing("fraction", Status.UNSUPPORTED),
            contact_slip=Quantity.missing("fraction", Status.UNSUPPORTED),
            sensor_health=Quantity.missing("fraction", Status.UNSUPPORTED), mode="driving",
            provenance=Provenance(origin="wheeled-ctrl-sim"))]
        if obs.goal_index != self._goal_index or t - self._goal_sent_s >= 1.0:
            self._goal_index, self._goal_sent_s = obs.goal_index, t
            progress = 1.0 - min(1.0, obs.goal_distance / 8.5)
            out.append(GoalState(header=self.emitter.header("goal", t), goal_id="waypoint", goal_type="navigate",
                                 priority=0.7, target=f"waypoint-{obs.goal_index}",
                                 progress=Quantity.measured(progress, "fraction")))
        live = set()
        for e in obs.entities:
            out.append(EntityObservation(
                header=self.emitter.header("entities", t), track_id=e.track, identity=e.identity,
                identity_confidence=e.identity_confidence if e.identity else 0.0, entity_type=e.kind,
                distance=Quantity.measured(e.distance, "m"), bearing=Quantity.measured(e.bearing, "rad"),
                closing_speed=Quantity.measured(e.closing, "m/s"), provenance=Provenance(origin="wheeled-ctrl-sim")))
            if e.ttc_p > 0 and e.closing > 0:  # D7: an estimate without closing motion is inconsistent; drop it
                live.add(e.track)
                out.append(OutcomeHypothesis(
                    header=self.emitter.header("hypotheses", t), hypothesis_id=f"collision:{e.track}",
                    kind="collision", target=e.track, affected="self", probability=e.ttc_p,
                    horizon_s=max(0.1, e.distance / e.closing), expected_magnitude=0.8, confidence=0.7,
                    provenance=Provenance(origin="wheeled-ctrl-adapter/ttc", method="estimated")))
        for track in sorted(self._live - live):  # D3: retract predictions whose condition has ended
            out.append(OutcomeHypothesis(
                header=self.emitter.header("hypotheses", t), hypothesis_id=f"collision:{track}", kind="collision",
                target=track, affected="self", probability=0.0, horizon_s=0.1, expected_magnitude=0.8,
                confidence=0.7, provenance=Provenance(origin="wheeled-ctrl-adapter/ttc", method="estimated")))
        self._live = live
        for event in obs.events:
            if event[0] == "collision":
                out.append(self._outcome(t, "collision", 1.0 if event[2] else 0.8, target=event[1]))
            elif event[0] == "slip":
                self._in_slip, self._calm_since = True, None
                out.append(self._outcome(t, "stability_loss", 0.6))
            elif event[0] == "goal_reached":
                out.append(self._outcome(t, "task_completed", 1.0, affected="waypoint"))
            elif event[0] == "goal_timeout":
                out.append(self._outcome(t, "task_interrupted", 0.6, affected="waypoint"))
        if self._in_slip:
            if obs.traction_margin >= 0.3:
                self._calm_since = t if self._calm_since is None else self._calm_since
                if t - self._calm_since >= 0.5:
                    self._in_slip = False
                    out.append(self._outcome(t, "recovery", 0.5))
            else:
                self._calm_since = None
        if not self._battery_warned and obs.energy < 0.1:
            self._battery_warned = True
            out.append(self._outcome(t, "resource_critical", 0.7))
        return out
