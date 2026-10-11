"""Outcome-grounded memory: attribution, ambiguous identities, no learning from predictions, persistence."""
from conftest import Robot


def encounter(robot, apm, t0, track, identity=None, confidence=0.0, outcome=None, hypothesis=False, seconds=2.0):
    t = t0
    while t < t0 + seconds:
        t += 0.1
        apm.ingest(robot.self_state(t))
        apm.ingest(robot.entity(t, track=track, identity=identity, confidence=confidence))
        if hypothesis:
            apm.ingest(robot.hypothesis(t, target=track, hid=f"h-{track}", p=0.9))
        apm.step(t)
    if outcome:
        apm.ingest(robot.outcome(t, kind=outcome, target=track, oid=f"o-{track}-{t0}"))
    apm.step(t + 0.05)
    apm.step(t + 3.0)  # track expires → encounter closes
    return t + 3.0


def test_outcomes_are_attributed_to_the_right_identity():
    robot = Robot()
    apm = robot.apm()
    t = 0.0
    for i in range(6):
        t = encounter(robot, apm, t, f"a{i}", identity="ann", confidence=1.0, outcome="collision")
        t = encounter(robot, apm, t, f"b{i}", identity="bob", confidence=1.0, outcome="benign_contact")
    ann, bob = apm.entities.expectation("ann", "person"), apm.entities.expectation("bob", "person")
    assert ann["p_adverse"] > 0.6 > bob["p_adverse"] and bob["p_benign"] > ann["p_benign"]
    assert ann["confidence"] > 0.6


def test_ambiguous_identity_gets_partial_credit():
    robot = Robot()
    sure, unsure = robot.apm(), robot.apm()
    t = 0.0
    for i in range(4):
        t = encounter(robot, sure, t, f"x{i}", identity="cara", confidence=1.0, outcome="collision")
    t2 = 0.0
    r2 = Robot()
    for i in range(4):
        t2 = encounter(r2, unsure, t2, f"x{i}", identity="cara", confidence=0.3, outcome="collision")
    assert sure.entities.records["id:cara"].encounters == 4.0
    assert abs(unsure.entities.records["id:cara"].encounters - 1.2) < 1e-9
    assert sure.entities.p("collision", "cara", "person")[1] > unsure.entities.p("collision", "cara", "person")[1]


def test_predictions_never_teach_outcomes():
    """A robot that keeps predicting collisions with someone who never collides learns that collisions are rare."""
    robot = Robot()
    apm = robot.apm()
    t = 0.0
    for i in range(8):
        t = encounter(robot, apm, t, f"d{i}", identity="dan", confidence=1.0, hypothesis=True)
    p, confidence = apm.entities.p("collision", "dan", "person")
    assert p < 0.05 and confidence > 0.7
    assert all(e.kind != "collision" for e in apm.episodes.episodes)


def test_undeclared_kinds_and_unattributable_outcomes():
    robot = Robot()
    apm = robot.apm()
    apm.ingest(robot.self_state(1.0))
    assert apm.ingest(robot.outcome(1.0, kind="fall", oid="f")) == "undeclared_outcome_kind"
    assert apm.ingest(robot.outcome(1.1, kind="collision", target="ghost", oid="g")) is None  # accepted...
    assert apm.entities.updates == 0 and "ghost" not in apm.entities.encounters               # ...not attributed


def test_persistence_roundtrip(tmp_path):
    robot = Robot()
    apm = robot.apm()
    t = 0.0
    for i in range(3):
        t = encounter(robot, apm, t, f"e{i}", identity="eve", confidence=1.0, outcome="collision")
    apm.save(tmp_path / "state.json")
    restored = type(apm).load(tmp_path / "state.json")
    assert restored.entities.expectation("eve", "person") == apm.entities.expectation("eve", "person")
    assert list(restored.episodes.episodes) == list(apm.episodes.episodes)
    assert (restored.affect.p == apm.affect.p).all()
    o = robot.outcome(t + 1, kind="collision", target=None, oid=apm.seen_outcomes[-1])
    assert restored.ingest(o, now_s=o.header.stamp_s) == "duplicate_outcome"


def test_memory_anticipates_without_any_external_prediction():
    """After repeated collisions with one identity, its next encounter is appraised as threatening before anything
    happens, through memory's own internal prediction; a stranger of the same type is appraised more mildly."""
    robot = Robot()
    apm = robot.apm()
    t = 0.0
    for i in range(5):
        t = encounter(robot, apm, t, f"f{i}", identity="fay", confidence=1.0, outcome="collision")
    for _ in range(300):  # five collisions in 25 s drive valence to its bound; let affect recover first
        t += 1.0
        apm.ingest(robot.self_state(t))
        apm.step(t)
    baseline = apm.affect.p.copy()
    assert baseline[0] > -0.2
    for _ in range(40):
        t += 0.1
        apm.ingest(robot.self_state(t))
        apm.ingest(robot.entity(t, track="again", identity="fay", confidence=1.0))
        out = apm.step(t)
    memory_frames = [f for f in out.appraisals if f.reference_type == "memory" and f.target == "again"]
    assert memory_frames and min(f.desirability for f in memory_frames) < 0
    assert apm.affect.p[0] < baseline[0] - 0.1
    assert next(x for x in out.tendencies if x.target == "again").avoid > 0.1
    updates_before = apm.entities.updates
    apm.step(t + 0.1)
    assert apm.entities.updates == updates_before  # memory's predictions are not evidence


def test_corrupt_or_foreign_state_is_never_used(tmp_path):
    from apm.core.runtime import load_or_new
    robot = Robot()
    apm = robot.apm()
    encounter(robot, apm, 0.0, "g0", identity="gus", confidence=1.0, outcome="collision")
    path = tmp_path / "state.json"
    apm.save(path)
    restored, reason = load_or_new(path, robot.profile)
    assert reason is None and restored.entities.updates == apm.entities.updates
    path.write_text(path.read_text()[:-40])  # truncated write
    fresh, reason = load_or_new(path, robot.profile)
    assert reason.startswith("corrupt") and fresh.entities.updates == 0
    other = Robot("otherbot")
    apm.save(path)
    fresh, reason = load_or_new(path, other.profile)
    assert reason == "state_from_another_embodiment"
