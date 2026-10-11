"""Wheeled-robot embodiment adapter (for the `apm_testbeds.wheeled` simulator) → AffectLink v1.

It provides its own capabilities and outcome semantics; it does not imitate the biped's signals:
  stability_margin   wheel traction margin 1 − |v·ω|/(μg), as a fraction clipped to [0, 1] (measured in simulation)
  energy_margin      battery fraction
  actuation_stress   |v| / v_max
  thermal, slip sensing, sensor health: UNSUPPORTED
Entities: obstacles and pedestrians within 3 m (track id, persistent identity for pedestrians with confidence 0.9,
distance, bearing, closing speed).
Outcome detectors:
  collision          new contact with an obstacle or pedestrian (target = its track). magnitude 0.8.
  stability_loss     wheel slip begins (traction margin < 0); re-arms when the margin exceeds 0.5. magnitude 0.6.
  recovery           after a slip, margin ≥ 0.3 continuously for 0.5 s. magnitude 0.5.
  task_completed     waypoint reached. magnitude 1.
  task_interrupted   no progress toward the waypoint (distance not reduced by 0.1 m) for 3 s. magnitude 0.6.
  resource_critical  battery falls below 10%, once. magnitude 0.7.
Prediction (OutcomeHypothesis): collision with a closing entity, p = σ((1.5 s − TTC)/0.4 s) for TTC < 3 s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from apm.adapters.base import Emitter
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

OUTCOMES = ("collision", "stability_loss", "recovery", "task_completed", "task_interrupted", "resource_critical")
V_MAX = 0.8


def profile(emitter: Emitter) -> EmbodimentProfile:
    return EmbodimentProfile(header=emitter.header("profile", 0.0), robot_id=emitter.source,
                             embodiment_class="wheeled", self_fields=("stability_margin", "energy_margin",
                                                                      "actuation_stress"),
                             outcome_kinds=OUTCOMES, goal_types=("navigate",))


@dataclass
class WheeledAdapter:
    emitter: Emitter
    _armed_slip: bool = True
    _in_slip: bool = False
    _calm_since: float | None = None
    _best_distance: float = math.inf
    _progress_since: float = 0.0
    _interrupted: bool = False
    _battery_warned: bool = False
    _goal: tuple[float, float] | None = None
    counter: int = 0
    _sent_goal_s: float = -1e9
    _tracks: set = field(default_factory=set)

    def profile(self) -> EmbodimentProfile:
        return profile(self.emitter)

    @staticmethod
    def _progress(s) -> float:
        return min(1.0, max(0.0, 1 - s.goal_distance / max(1e-6, s.goal_start_distance)))

    def _outcome(self, t, kind, magnitude, target=None, affected="self") -> OutcomeSignal:
        self.counter += 1
        return OutcomeSignal(header=self.emitter.header("outcomes", t),
                             outcome_id=f"{self.emitter.source}:{self.counter}",
                             kind=kind, magnitude=magnitude, target=target, affected=affected, evidence_quality=1.0,
                             confirmed=True, provenance=Provenance(origin="wheeled-adapter/detector"))

    def step(self, s) -> list:
        t = s.t
        out: list = [SelfState(
            header=self.emitter.header("self_state", t),
            stability_margin=Quantity.measured(min(1.0, max(0.0, s.traction_margin)), "fraction"),
            energy_margin=Quantity.measured(s.energy, "fraction"),
            actuation_stress=Quantity.measured(min(1.0, abs(s.v) / V_MAX), "fraction"),
            thermal_margin=Quantity.missing("fraction", Status.UNSUPPORTED),
            contact_slip=Quantity.missing("fraction", Status.UNSUPPORTED),
            sensor_health=Quantity.missing("fraction", Status.UNSUPPORTED), mode="driving",
            provenance=Provenance(origin="wheeled-sim"))]
        if s.goal != self._goal or t - self._sent_goal_s >= 1.0:
            if s.goal != self._goal:
                self._goal, self._best_distance, self._progress_since, self._interrupted = s.goal, math.inf, t, False
            self._sent_goal_s = t
            out.append(GoalState(header=self.emitter.header("goal", t), goal_id="waypoint", goal_type="navigate",
                                 priority=0.7, target=f"({s.goal[0]:.2f},{s.goal[1]:.2f})",
                                 progress=Quantity.measured(self._progress(s), "fraction")))
        for e in s.entities:
            out.append(EntityObservation(
                header=self.emitter.header("entities", t), track_id=e["track"], identity=e["identity"],
                identity_confidence=0.9 if e["identity"] else 0.0, entity_type=e["kind"],
                distance=Quantity.measured(e["distance"], "m"), bearing=Quantity.measured(e["bearing"], "rad"),
                closing_speed=Quantity.measured(e["closing"], "m/s"), provenance=Provenance(origin="wheeled-sim")))
            if e["closing"] > 0.05:
                ttc = e["distance"] / e["closing"]
                if ttc < 3.0:
                    p = 1 / (1 + math.exp(-(1.5 - ttc) / 0.4))
                    out.append(OutcomeHypothesis(
                        header=self.emitter.header("hypotheses", t), hypothesis_id=f"collision:{e['track']}",
                        kind="collision", target=e["track"], affected="self", probability=p, horizon_s=max(0.1, ttc),
                        expected_magnitude=0.8, confidence=0.7,
                        provenance=Provenance(origin="wheeled-adapter/ttc", method="estimated")))
        for c in s.collisions:
            out.append(self._outcome(t, "collision", 0.8, target=c))
        if self._armed_slip and s.traction_margin < 0:
            self._armed_slip, self._in_slip, self._calm_since = False, True, None
            out.append(self._outcome(t, "stability_loss", 0.6))
        if self._in_slip:
            if s.traction_margin >= 0.3:
                self._calm_since = t if self._calm_since is None else self._calm_since
                if t - self._calm_since >= 0.5:
                    self._in_slip = False
                    out.append(self._outcome(t, "recovery", 0.5))
            else:
                self._calm_since = None
        if not self._armed_slip and s.traction_margin > 0.5:
            self._armed_slip = True
        if s.reached:
            out.append(self._outcome(t, "task_completed", 1.0, affected="waypoint"))
        elif s.goal_distance < self._best_distance - 0.1:
            self._best_distance, self._progress_since, self._interrupted = s.goal_distance, t, False
        elif not self._interrupted and t - self._progress_since >= 3.0:
            self._interrupted = True
            out.append(self._outcome(t, "task_interrupted", 0.6, affected="waypoint"))
        if not self._battery_warned and s.energy < 0.1:
            self._battery_warned = True
            out.append(self._outcome(t, "resource_critical", 0.7))
        return out
