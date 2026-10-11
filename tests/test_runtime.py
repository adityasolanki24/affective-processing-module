"""Runtime outputs: target-specific tendencies, attention, expression bounds, diagnostics."""
from conftest import Robot

from apm.protocol import decode, encode


def test_tendencies_are_target_specific():
    robot = Robot()
    apm = robot.apm()
    for i in range(40):
        t = 0.05 * (i + 1)
        apm.ingest(robot.self_state(t))
        apm.ingest(robot.goal(t))
        apm.ingest(robot.entity(t, track="hazard", distance=0.5, closing=1.0))
        apm.ingest(robot.entity(t, track="dock", entity_type="object"))
        apm.ingest(robot.hypothesis(t, kind="collision", target="hazard", hid="h1", p=0.8, horizon=2.0))
        apm.ingest(robot.hypothesis(t, kind="task_completed", target="dock", hid="h2", p=0.7, affected="g1"))
        out = apm.step(t)
    by = {x.target: x for x in out.tendencies}
    assert by["hazard"].avoid > 0.3 and by["hazard"].approach == 0.0
    assert by["dock"].approach > 0.3 and by["dock"].avoid == 0.0
    assert out.attention[0].target in ("hazard", "dock")


def test_low_control_self_strain_seeks_assistance():
    robot = Robot()
    apm = robot.apm()
    for i in range(40):
        t = 0.05 * (i + 1)
        apm.ingest(robot.self_state(t, stability=0.05))
        apm.ingest(robot.hypothesis(t, kind="stability_loss", target=None, p=0.9, horizon=0.2))
        out = apm.step(t)
    me = next(x for x in out.tendencies if x.target == "self")
    assert me.seek_assistance > 0.5 and me.disengage > me.avoid


def test_outputs_are_valid_affectlink_messages_and_bounded():
    robot = Robot()
    apm = robot.apm()
    apm.ingest(robot.self_state(1.0))
    apm.ingest(robot.entity(1.0))
    apm.ingest(robot.hypothesis(1.0))
    out = apm.step(1.0)
    for m in out.messages():
        assert decode(encode(m)) == m
    e = out.expression
    assert 0 <= e.activation <= 1 and -1 <= e.posture <= 1 and 0 <= e.movement_energy <= 1 and 0 <= e.urgency <= 1


def test_diagnostics_report_stale_inputs():
    robot = Robot()
    apm = robot.apm()
    apm.ingest(robot.self_state(1.0))
    assert apm.step(1.5).diagnostics.health == "ok"
    late = apm.step(10.0).diagnostics
    assert late.health == "stale_inputs" and late.input_age_s["self_state"] == 9.0


def test_a_robot_that_stays_down_keeps_seeking_assistance():
    robot = Robot()
    apm = robot.apm()
    apm.ingest(robot.self_state(0.1, stability=0.9))
    apm.ingest(robot.outcome(0.1, kind="stability_loss", target=None, oid="fall"))
    for i in range(200):  # 20 s after the outcome frame expired, margin still zero
        t = 0.2 + 0.1 * i
        apm.ingest(robot.self_state(t, stability=0.0))
        out = apm.step(t)
    assert next(x for x in out.tendencies if x.target == "self").seek_assistance > 0.9
