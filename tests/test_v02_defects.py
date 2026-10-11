"""APM v0.2 audit: suspected v0.1 defects, each verified by a test before any fix (experiments/F_behavioral_learning/
AUDIT.md). Tests that pin a v0.1 behaviour which is *kept* (because changing it would alter the validated baseline)
say so and assert the documented behaviour instead."""
import math

import numpy as np
from conftest import Robot

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.wheeled import WheeledAdapter
from apm.core.context import MAX_ENTITIES
from apm.protocol.messages import OutcomeHypothesis, Reset
from apm_testbeds.wheeled import WheeledWorld


def test_d1_reset_ends_open_encounters():
    """D1: a sender Reset replaced the context but left the entity encounters open forever, so their evidence was
    never credited and a reused track id silently merged two encounters across the restart."""
    robot = Robot()
    apm = robot.apm()
    for i in range(10):
        t = 0.1 * (i + 1)
        apm.ingest(robot.self_state(t))
        apm.ingest(robot.entity(t, track="t1", identity="ann", confidence=1.0))
        apm.step(t)
    apm.ingest(robot.outcome(1.0, kind="collision", target="t1", oid="c1"))
    robot.emit.restart()
    assert apm.ingest(Reset(header=robot.emit.header("control", 1.1), reason="restart")) is None
    apm.step(1.2)
    assert "t1" not in apm.entities.encounters
    assert apm.entities.records["id:ann"].counts["collision"] == 1.0


def test_d2_evicted_tracks_end_their_encounters():
    """D2: when the bounded context evicted its oldest track, the matching encounter was never ended (a leak)."""
    robot = Robot()
    apm = robot.apm()
    t = 0.0
    for i in range(MAX_ENTITIES + 6):
        t += 0.001
        apm.ingest(robot.entity(t, track=f"k{i}"), now_s=t)
    assert len(apm.entities.encounters) <= MAX_ENTITIES
    assert apm.entities.records["type:person"].encounters == 6.0


def run_wheeled(seed, steps=1200):
    world, adapter = WheeledWorld(seed), WheeledAdapter(Emitter("w"))
    apm = APM(profile=adapter.profile())
    log = []
    for _ in range(steps):
        tel = world.step()
        msgs = adapter.step(tel)
        for m in msgs:
            apm.ingest(m)
        apm.step(tel.t)
        log.append((tel, msgs, dict(apm.context.hypotheses)))
    return log


def test_d3_v01_ttc_hypotheses_go_stale_when_closing_stops():
    """D3 (v0.1 adapter, kept unchanged): a TTC collision hypothesis is never retracted. When the entity stops
    closing, the APM keeps the last probability for up to its horizon (≤ 3 s). This test documents the stale window
    in the v0.1 wheeled adapter; the v0.2 controllable adapter retracts (tests/test_wheeled_ctrl.py)."""
    stale = 0
    for seed in range(4):
        for tel, _msgs, held in run_wheeled(70_000 + seed, 600):
            closing = {e["track"] for e in tel.entities if e["closing"] > 0.05}
            stale += sum(1 for h in held.values() if h.kind == "collision" and h.target not in closing
                         and h.probability > 0.2)
    assert stale > 0  # verified defect: held collision predictions for entities that are no longer closing


def test_d4_v01_collision_detector_double_counts_contact_chatter():
    """D4 (v0.1 testbed, kept unchanged): the robot is pushed back on contact; contact can break for one step and
    resume, which the detector reports as a new collision with the same entity within 0.5 s."""
    repeats = 0
    for seed in range(20):
        world = WheeledWorld(50_000 + seed)
        last: dict[str, float] = {}
        for _ in range(1200):
            tel = world.step()
            for c in tel.collisions:
                repeats += c in last and tel.t - last[c] < 0.5
                last[c] = tel.t
    assert repeats > 0


def test_d5_v01_world_rng_couples_physics_and_tasks():
    """D5 (v0.1 testbed, kept unchanged): one RNG drives layout, slip noise and new goals, so anything that changes
    the robot's trajectory also changes every later goal: paired comparisons are not common-random-number paired."""
    a, b = WheeledWorld(11), WheeledWorld(11)
    b.rng.normal()  # one extra draw, as a single slip would cause
    a._new_goal()
    b._new_goal()
    assert a.goal != b.goal


def test_d6_v01_ttc_estimator_fires_on_most_routine_approaches():
    """D6 (measurement): the v0.1 TTC estimator raises p ≥ 0.5 on a large fraction of control steps, and most of
    those alarms are not followed by a collision within 1 s (AUDIT.md reports the measured rates)."""
    alarms, hits, steps = 0, 0, 0
    for seed in range(6):
        log = run_wheeled(50_000 + seed)
        collisions = np.array([tel.t for tel, _, _ in log for _ in tel.collisions])
        for tel, msgs, _ in log:
            steps += 1
            ps = [m.probability for m in msgs if isinstance(m, OutcomeHypothesis)]
            if ps and max(ps) >= 0.5:
                alarms += 1
                hits += bool(np.any((collisions > tel.t) & (collisions <= tel.t + 1.0)))
    assert alarms / steps > 0.1 and hits / alarms < 0.5
    assert math.isfinite(hits / alarms)
