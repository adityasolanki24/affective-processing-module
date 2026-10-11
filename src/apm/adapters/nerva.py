"""NERVA embodiment adapter: Open Duck Mini v2 biped telemetry (MuJoCo simulation) → AffectLink v1.

Consumes per-control-step telemetry as plain arrays, the fields NERVA's simulator logs (`nerva.sim.open_duck.StepLog`):
t, base_quat (w, x, y, z), base_linvel (world), base_angvel, tau_sq (mean Σ actuator torque² per substep), plus
the active locomotion command (vx, vy, yaw_rate). It never imports NERVA, so APM stays independent of it.

Signals actually available, and how they are mapped (fixed before any APM learning experiment):
  stability_margin   torso tilt (deg) via calibration nominal 0° → critical 45° (NERVA's motor gate counts > 45° as
                     a fall). Measured.
  actuation_stress   RMS actuator torque / 3.23 N·m (the sts3215 force range in NERVA's backlash model). Measured.
  energy, thermal, contact slip, sensor health: not simulated → UNSUPPORTED (never zero).
Outcome detectors (OutcomeSignal, measured from simulator state):
  stability_loss     tilt crosses 25° (NERVA's SafetySupervisor stop threshold); re-arms below 12°.
                     magnitude = clip((peak tilt − 25) / 20, 0.2, 1).
  fall               tilt > 45° or a non-finite state. magnitude 1.
  recovery           after a stability_loss, tilt ≤ 12° continuously for 0.5 s (NERVA's resume rule). magnitude 0.6.
  task_interrupted   while a moving command is active: the 1 s mean velocity along the commanded axis stays below 25%
                     of the command for 1 s; re-arms after it recovers above 50%. magnitude = 1 − ratio (clipped).
Prediction (OutcomeHypothesis, an estimate, never learned as an outcome):
  stability_loss     tilt extrapolated 0.3 s ahead with the measured tilt rate; p = σ((tilt_pred − 25°)/3°).
                     Emitted only when p ≥ 0.05. Provenance "nerva-adapter/tilt-extrapolation".
No entity, collision or contact-impact signals exist in the stored NERVA motor telemetry, so none are emitted.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from apm.adapters.base import Emitter
from apm.protocol.messages import (
    EmbodimentProfile,
    GoalState,
    MarginCalibration,
    OutcomeHypothesis,
    OutcomeSignal,
    Provenance,
    Quantity,
    SelfState,
    Status,
)

FALL_DEG, LOSS_DEG, REARM_DEG, RECOVER_HOLD_S = 45.0, 25.0, 12.0, 0.5
MAX_TORQUE_NM, ACTUATORS = 3.23, 14
INTERRUPT_RATIO, RESUME_RATIO, INTERRUPT_HOLD_S, VELOCITY_WINDOW_S = 0.25, 0.5, 1.0, 1.0
PREDICT_AHEAD_S, PREDICT_SCALE_DEG, PREDICT_MIN_P = 0.3, 3.0, 0.05
OUTCOMES = ("stability_loss", "fall", "recovery", "task_interrupted")


def tilt_deg(q) -> float:
    w, x, y, z = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def yaw_rad(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def profile(emitter: Emitter) -> EmbodimentProfile:
    return EmbodimentProfile(
        header=emitter.header("profile", 0.0), robot_id=emitter.source, embodiment_class="biped",
        self_fields=("stability_margin", "actuation_stress"), outcome_kinds=OUTCOMES,
        goal_types=("track_velocity", "stand"),
        calibration=(MarginCalibration(field="stability_margin", raw_unit="deg", nominal=0.0, critical=FALL_DEG,
                                       meaning="torso tilt from vertical; > 45 deg is a fall in NERVA's motor gate"),))


@dataclass
class NervaAdapter:
    emitter: Emitter
    goal_every_s: float = 1.0
    _t_prev: float | None = None
    _tilt_prev: float | None = None
    _armed_loss: bool = True
    _in_loss: bool = False
    _peak: float = 0.0
    _calm_since: float | None = None
    _fallen: bool = False
    _interrupt_since: float | None = None
    _interrupted: bool = False
    _goal_sent_s: float = -1e9
    _command: tuple[float, float, float] | None = None
    _velocity: deque = field(default_factory=deque)
    counter: int = 0

    def profile(self) -> EmbodimentProfile:
        return profile(self.emitter)

    def step(self, t: float, base_quat, base_linvel, base_angvel, tau_sq: float,
             command: tuple[float, float, float]) -> list:
        """One control step of telemetry → the AffectLink messages it implies (self state, goal, outcomes,
        hypothesis)."""
        messages: list = []
        finite = all(np.isfinite(np.asarray(v, dtype=float)).all() for v in (base_quat, base_linvel, base_angvel))
        tilt = tilt_deg(base_quat) if finite else float("nan")
        goal = self._goal(t, command)  # updates the active command first, so the reported mode matches it
        messages.append(self._self_state(t, tilt, tau_sq, finite))
        messages += goal
        messages += self._stability(t, tilt, finite)
        messages += self._task(t, base_quat, base_linvel, base_angvel, command, finite)
        messages += self._predict(t, tilt, finite)
        self._t_prev, self._tilt_prev = t, tilt if finite else None
        return messages

    # -------------------------------------------------- self state and goal
    def _self_state(self, t: float, tilt: float, tau_sq: float, finite: bool) -> SelfState:
        stress = math.sqrt(max(0.0, tau_sq) / ACTUATORS) / MAX_TORQUE_NM if math.isfinite(tau_sq) else None
        return SelfState(
            header=self.emitter.header("self_state", t),
            stability_margin=Quantity.measured(tilt, "deg") if finite else Quantity.missing("deg", Status.INVALID),
            actuation_stress=Quantity.measured(min(1.0, stress), "fraction") if stress is not None
            else Quantity.missing("fraction", Status.INVALID),
            energy_margin=Quantity.missing("fraction", Status.UNSUPPORTED),
            thermal_margin=Quantity.missing("fraction", Status.UNSUPPORTED),
            contact_slip=Quantity.missing("fraction", Status.UNSUPPORTED),
            sensor_health=Quantity.missing("fraction", Status.UNSUPPORTED),
            mode="fallen" if self._fallen else ("standing" if not any(command_nonzero(self._command)) else "walking"),
            provenance=Provenance(origin="nerva-mujoco", method="measured"))

    def _goal(self, t: float, command) -> list:
        command = tuple(float(c) for c in command)
        changed = command != self._command
        if changed:
            self._velocity.clear()
            self._interrupt_since, self._interrupted = None, False
        self._command = command
        if not changed and t - self._goal_sent_s < self.goal_every_s:
            return []
        self._goal_sent_s = t
        moving = any(command_nonzero(command))
        return [GoalState(header=self.emitter.header("goal", t), goal_id="locomotion",
                          goal_type="track_velocity" if moving else "stand", priority=0.6,
                          target=f"cmd({command[0]:+.3f},{command[1]:+.3f},{command[2]:+.2f})",
                          progress=Quantity.missing("fraction", Status.UNSUPPORTED), active=True)]

    # -------------------------------------------------- outcome detectors
    def _outcome(self, t: float, kind: str, magnitude: float, affected: str) -> OutcomeSignal:
        self.counter += 1
        return OutcomeSignal(header=self.emitter.header("outcomes", t),
                             outcome_id=f"{self.emitter.source}:{kind}:{self.counter}",
                             kind=kind, target=None, affected=affected, magnitude=float(min(1.0, max(0.0, magnitude))),
                             evidence_quality=1.0, confirmed=True,
                             provenance=Provenance(origin="nerva-adapter/detector", method="measured"))

    def _stability(self, t: float, tilt: float, finite: bool) -> list:
        out = []
        if not self._fallen and (not finite or tilt > FALL_DEG):
            self._fallen = True
            out.append(self._outcome(t, "fall", 1.0, "self"))
            return out
        if self._fallen or not finite:
            return out
        if self._armed_loss and tilt >= LOSS_DEG:
            self._armed_loss, self._in_loss, self._peak, self._calm_since = False, True, tilt, None
            out.append(self._outcome(t, "stability_loss", max(0.2, (tilt - LOSS_DEG) / 20.0), "self"))
        if self._in_loss:
            self._peak = max(self._peak, tilt)
            if tilt <= REARM_DEG:
                self._calm_since = t if self._calm_since is None else self._calm_since
                if t - self._calm_since >= RECOVER_HOLD_S:
                    self._in_loss = False
                    out.append(self._outcome(t, "recovery", 0.6, "self"))
            else:
                self._calm_since = None
        if not self._armed_loss and tilt < REARM_DEG:
            self._armed_loss = True
        return out

    def _task(self, t: float, q, v, w, command, finite: bool) -> list:
        if not finite or self._fallen or not any(command_nonzero(command)):
            return []
        axis = int(np.argmax(np.abs(command)))
        yaw = yaw_rad(q)
        c, s = math.cos(yaw), math.sin(yaw)
        heading = (c * v[0] + s * v[1], -s * v[0] + c * v[1], w[2])
        self._velocity.append((t, heading[axis]))
        while self._velocity and t - self._velocity[0][0] > VELOCITY_WINDOW_S:
            self._velocity.popleft()
        if t - self._velocity[0][0] < VELOCITY_WINDOW_S * 0.9:
            return []  # not a full window yet
        ratio = float(np.mean([x for _, x in self._velocity])) / command[axis]
        if not self._interrupted:
            if ratio < INTERRUPT_RATIO:
                self._interrupt_since = t if self._interrupt_since is None else self._interrupt_since
                if t - self._interrupt_since >= INTERRUPT_HOLD_S:
                    self._interrupted = True
                    return [self._outcome(t, "task_interrupted", 1.0 - max(0.0, ratio), "locomotion")]
            else:
                self._interrupt_since = None
        elif ratio > RESUME_RATIO:
            self._interrupted, self._interrupt_since = False, None
        return []

    # -------------------------------------------------- prediction
    def _predict(self, t: float, tilt: float, finite: bool) -> list:
        if not finite or self._fallen or self._t_prev is None or self._tilt_prev is None or t <= self._t_prev:
            return []
        rate = (tilt - self._tilt_prev) / (t - self._t_prev)
        predicted = tilt + rate * PREDICT_AHEAD_S
        p = 1.0 / (1.0 + math.exp(-(predicted - LOSS_DEG) / PREDICT_SCALE_DEG))
        if p < PREDICT_MIN_P or not self._armed_loss:
            return []
        return [OutcomeHypothesis(header=self.emitter.header("hypotheses", t), hypothesis_id="nerva:stability_loss",
                                  kind="stability_loss", target=None, affected="self", probability=float(p),
                                  horizon_s=PREDICT_AHEAD_S, expected_magnitude=0.5, confidence=0.6,
                                  provenance=Provenance(origin="nerva-adapter/tilt-extrapolation",
                                                        method="estimated"))]


def command_nonzero(command) -> tuple[bool, ...]:
    return tuple(abs(c) > 1e-9 for c in (command or (0.0, 0.0, 0.0)))


def replay(arrays: dict, commands, emitter: Emitter):
    """Yield (t, messages) for a stored NERVA rollout. `commands`: one (vx, vy, yaw_rate) per control step."""
    adapter = NervaAdapter(emitter)
    yield 0.0, [adapter.profile()]
    for i in range(len(arrays["t"])):
        t = float(arrays["t"][i])
        yield t, adapter.step(t, arrays["base_quat"][i], arrays["base_linvel"][i], arrays["base_angvel"][i],
                              float(arrays["tau_sq"][i]), tuple(commands[i]))
