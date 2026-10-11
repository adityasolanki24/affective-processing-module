"""NERVA adapter detectors and mappings on synthetic trajectories (the real-data run is examples/nerva/)."""
import math

import numpy as np

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.nerva import NervaAdapter, tilt_deg
from apm.protocol import validate
from apm.protocol.messages import OutcomeHypothesis, OutcomeSignal, SelfState, Status


def quat_tilt(deg):
    a = math.radians(deg) / 2
    return np.array([math.cos(a), math.sin(a), 0.0, 0.0])  # roll by `deg`


def drive(tilts, command=(0.074, 0.0, 0.0), speed=0.074, dt=0.02):
    adapter = NervaAdapter(Emitter("nerva-test"))
    msgs = [adapter.profile()]
    for i, tilt in enumerate(tilts):
        t = (i + 1) * dt
        msgs += adapter.step(t, quat_tilt(tilt), np.array([speed, 0.0, 0.0]), np.zeros(3), 14 * 1.0 ** 2, command)
    return msgs


def kinds(msgs):
    return [m.kind for m in msgs if isinstance(m, OutcomeSignal)]


def test_mappings_are_measured_or_unsupported_never_zero():
    msgs = drive([5.0] * 10)
    s = next(m for m in msgs if isinstance(m, SelfState))
    assert abs(s.stability_margin.value - 5.0) < 1e-9 and s.stability_margin.unit == "deg"
    assert s.mode == "walking"
    assert abs(s.actuation_stress.value - 1.0 / 3.23) < 1e-9
    assert s.energy_margin.status is Status.UNSUPPORTED and s.energy_margin.value is None
    for m in msgs:
        validate(m)
    apm = APM(profile=msgs[0])
    apm.ingest(s)
    assert abs(apm.context.margin("stability_margin", s.header.stamp_s) - (1 - 5 / 45)) < 1e-9


def test_stability_loss_recovery_and_rearm():
    tilts = [5] * 10 + list(np.linspace(5, 30, 20)) + [30] * 5 + list(np.linspace(30, 5, 20)) + [5] * 40
    msgs = drive(tilts)
    assert kinds(msgs) == ["stability_loss", "recovery"]
    assert kinds(drive(tilts + tilts)) == ["stability_loss", "recovery"] * 2


def test_fall_is_reported_once_and_ends_detection():
    msgs = drive([5] * 5 + list(np.linspace(5, 60, 30)) + [80] * 20)
    assert kinds(msgs) == ["stability_loss", "fall"]


def test_task_interruption_needs_a_sustained_shortfall():
    adapter = NervaAdapter(Emitter("n"))
    out = []
    for i in range(300):
        t = (i + 1) * 0.02
        speed = 0.074 if t < 2 or t > 5 else 0.0
        out += adapter.step(t, quat_tilt(2.0), np.array([speed, 0, 0]), np.zeros(3), 1.0, (0.074, 0, 0))
    assert kinds(out) == ["task_interrupted"]


def test_prediction_is_a_hypothesis_not_an_outcome():
    msgs = drive([5] * 5 + list(np.linspace(5, 22, 10)))
    hyps = [m for m in msgs if isinstance(m, OutcomeHypothesis)]
    assert hyps and all(m.kind == "stability_loss" and m.provenance.method == "estimated" for m in hyps)
    assert "stability_loss" not in kinds(msgs)  # tilt never reached 25°


def test_tilt_helper():
    assert abs(tilt_deg(quat_tilt(30.0)) - 30.0) < 1e-9
