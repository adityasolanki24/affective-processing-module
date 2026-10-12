"""Closed-loop expressive episodes: intent schedule -> adapter -> controller, with the independent monitor.

Works with any controller exposing the small interface below: `apm_express.biped.NervaBiped` (MuJoCo, NERVA's
validated policy) or `apm_express.kinematic.KinematicBiped` (a kinematic stand-in for fast tests). The monitor
only sees `controller.physical()`; when it trips, the neutral request is sent and the adapter is told to restart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from apm_express.adapter import COMMAND_NAMES, NEUTRAL_REQUEST, BipedExpressionAdapter, MotionRequest
from apm_express.intent import ExpressionIntent, blend
from apm_express.safety import SafetyMonitor


class Controller(Protocol):
    t: float
    dt: float

    def pose(self) -> tuple[float, float, float]: ...
    def physical(self) -> dict: ...
    def apply(self, request: MotionRequest) -> None: ...
    def advance(self) -> dict: ...
    def push(self, dvx: float, dvy: float) -> None: ...


@dataclass
class EpisodeSpec:
    seconds: float
    target: tuple[float, float] | None
    intents: list[tuple[float, ExpressionIntent]]  # (start time, intent), piecewise constant, sorted
    pushes: dict[int, tuple[float, float]] = field(default_factory=dict)  # control step -> world (dvx, dvy)
    blend_s: float = 0.0  # linear transition between consecutive intents (0 = step change)
    override: list[tuple[float, float, MotionRequest]] = field(default_factory=list)  # (t0, t1, raw request): H0 tests

    def intent_at(self, t: float) -> ExpressionIntent:
        current = self.intents[0][1]
        for i, (start, intent) in enumerate(self.intents):
            if t + 1e-9 < start:
                break
            previous = self.intents[i - 1][1] if i > 0 else intent
            w = 1.0 if self.blend_s <= 0 else (t - start) / self.blend_s
            current = blend(previous, intent, w) if i > 0 and w < 1 else intent
        return current

    def override_at(self, t: float) -> MotionRequest | None:
        for t0, t1, req in self.override:
            if t0 - 1e-9 <= t < t1 - 1e-9:
                return req
        return None


def run_episode(ctrl: Controller, adapter: BipedExpressionAdapter, monitor: SafetyMonitor, spec: EpisodeSpec,
                record_rows: bool = True, enforce: bool = True) -> dict:
    """Run one episode; returns per-step arrays plus events. Never raises on instability: it is recorded.

    enforce=False runs the monitor in shadow (trips are recorded, requests pass unchanged): used only by H0 to
    measure what the controller does without the fallback.
    """
    n = int(round(spec.seconds / ctrl.dt))
    adapter.reset(ctrl.t, ctrl.pose(), spec.target, spec.intents[0][1])
    rows: list[dict] = []
    fallback_steps = 0
    completed = True
    for k in range(n):
        if k in spec.pushes:
            ctrl.push(*spec.pushes[k])
        t = ctrl.t
        phys = ctrl.physical()
        tripped = monitor.update(t, **phys)
        if tripped and enforce:
            request = NEUTRAL_REQUEST
            fallback_steps += 1
            adapter.notify_fallback(t)
        else:
            request = spec.override_at(t) or adapter.request(t, ctrl.pose(), spec.target, spec.intent_at(t))
        if request.command not in COMMAND_NAMES:
            raise ValueError("adapter produced an unknown command")
        ctrl.apply(request)
        row = ctrl.advance()
        if record_rows:
            row.update(command=COMMAND_NAMES.index(request.command), e_pitch=request.e_pitch,
                       head_request=np.array(request.head), fallback=tripped)
            rows.append(row)
        if not row.get("finite", True):
            completed = False
            break
    arrays = {key: np.array([r[key] for r in rows]) for key in rows[0]} if rows else {}
    return {"arrays": arrays, "completed": completed, "fallback_steps": fallback_steps, "trips": monitor.trips,
            "trip_reasons": list(monitor.reasons), "min_joint_margin": monitor.min_joint_margin,
            "max_saturated_run": monitor.max_saturated_run, "dt": ctrl.dt}
