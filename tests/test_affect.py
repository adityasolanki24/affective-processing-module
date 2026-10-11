"""Affect dynamics: convergence (no accumulation), recovery, rate invariance, missing data, phasic events."""
import numpy as np
from conftest import Robot


def run(robot: Robot, hz: float, seconds: float, hypothesis=True, start=0.0, apm=None, stability=0.9):
    apm = apm or robot.apm()
    dt = 1.0 / hz
    for i in range(int(round(seconds * hz))):
        t = start + (i + 1) * dt
        apm.ingest(robot.self_state(t, stability=stability))
        apm.ingest(robot.entity(t, distance=0.6, closing=0.8))
        if hypothesis:
            apm.ingest(robot.hypothesis(t, p=0.8, horizon=1.0))
        apm.step(t)
    return apm


def test_sustained_context_converges_without_saturating():
    robot = Robot()
    apm = run(robot, 50, 300)
    p300 = apm.affect.p.copy()
    run(robot, 50, 300, apm=apm, start=300)
    assert np.allclose(apm.affect.p, p300, atol=1e-4)  # converged: repetition adds nothing
    assert np.all(np.abs(apm.affect.p) < 0.9)          # bounded away from saturation
    assert max(apm.affect.saturation().values()) == 0.0
    assert apm.affect.p[0] < -0.2 and apm.affect.p[1] > 0.2  # an adverse prediction lowers V, raises A


def test_recovery_when_context_clears():
    robot = Robot()
    apm = run(robot, 20, 60)
    threatened = apm.affect.p.copy()
    run(robot, 20, 600, hypothesis=False, apm=apm, start=60)
    assert np.linalg.norm(apm.affect.p) < 0.2 * np.linalg.norm(threatened)


def test_rate_invariance_10_vs_100_hz():
    a = run(Robot(), 10, 30).affect.p
    b = run(Robot(), 100, 30).affect.p
    assert np.allclose(a, b, atol=1e-3)


def test_phasic_outcome_counts_once_and_decays():
    robot = Robot()
    apm = run(robot, 20, 5, hypothesis=False)
    before = apm.affect.p.copy()
    o = robot.outcome(5.01, kind="collision", target="t1", magnitude=1.0)
    assert apm.ingest(o) is None
    assert apm.ingest(o) is not None  # exact duplicate (same seq) rejected
    dup = robot.outcome(5.02, kind="collision", target="t1", magnitude=1.0, oid="o1")
    assert apm.ingest(dup) == "duplicate_outcome"  # resent outcome id: counted once
    apm.step(5.5)
    peak = apm.affect.p.copy()
    assert peak[0] < before[0] - 0.05 and peak[1] > before[1] + 0.05
    run(robot, 20, 120, hypothesis=False, apm=apm, start=5.5)
    assert abs(apm.affect.p[0] - before[0]) < 0.25 * abs(peak[0] - before[0])


def test_missing_margins_are_not_treated_as_zero_or_full():
    robot = Robot()
    apm = robot.apm()
    apm.ingest(robot.self_state(1.0, stability=None, energy_v=None))
    out = apm.step(1.0)
    assert apm.context.body_margins(1.0) == {}
    assert set(out.diagnostics.missing_capabilities) == {"stability_margin", "energy_margin"}
    assert out.diagnostics.health == "degraded"
    assert apm.affect.u[2] == 0.0 and apm.affect.u[4] > 0.0  # no fabricated strain; uncertainty rises instead


def test_raw_values_need_their_calibration():
    robot = Robot()
    apm = robot.apm()
    apm.ingest(robot.self_state(1.0, energy_v=11.55))
    assert abs(apm.context.margin("energy_margin", 1.0) - 0.5) < 1e-9  # (11.55 − 10.5) / (12.6 − 10.5)
    assert apm.context.margin("thermal_margin", 1.0) is None


def test_low_margin_raises_relevance_and_strain():
    calm = run(Robot(), 20, 60, stability=0.9).affect.p
    strained = run(Robot(), 20, 60, stability=0.1).affect.p
    assert strained[0] < calm[0] and strained[1] > calm[1]
