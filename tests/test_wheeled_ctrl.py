"""E0 gate: the v0.2 intervention-capable testbed supports fair, causal comparisons (MASTER_PREREGISTRATION.md E0)."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.wheeled_ctrl import WheeledCtrlAdapter
from apm.protocol import validate
from apm.protocol.messages import OutcomeHypothesis, OutcomeSignal
from apm_research.dataset import DecisionRecorder
from apm_research.features import WIDTH
from apm_research.harness import Lifetime, run_episode, run_lifetime
from apm_research.memory import ObsMemory
from apm_research.policies import B0, B2, Explore, Policy
from apm_research.predictor import ActionPredictor
from apm_testbeds.wheeled_ctrl import (
    ACTIONS,
    DECISION_STEPS,
    GO,
    SLOW,
    YIELD,
    Identity,
    Obs,
    Seen,
    WheeledCtrlWorld,
    child,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "F_behavioral_learning"))
import seeds as seedmod  # noqa: E402

POP = [Identity(f"ped-{i}", "careless" if i % 2 else "careful") for i in range(6)]


def world(i=0, pop=POP):
    return WheeledCtrlWorld(seedmod.world_seed("E0", "P", i), pop)


def drive(w, actions):
    out = []
    for a in actions:
        for _ in range(DECISION_STEPS):
            out.append(w.step(a))
    return out


# ---------------------------------------------------------------- 1. interventions are causal
def test_actions_change_subsequent_motion_and_outcomes():
    a, b = world(1), world(1)
    drive(a, [GO] * 4)
    drive(b, [YIELD] * 4)
    assert np.hypot(a.x - b.x, a.y - b.y) > 0.3
    rates = {}
    for action in (GO, SLOW):
        cols = goals = 0
        for i in range(12):
            w = world(i)
            while not w.done:
                w.step(action)
            cols += len(w.log.collisions)
            goals += w.log.goals_completed
        rates[action] = (cols, goals)
    assert rates[SLOW][0] < rates[GO][0] and rates[SLOW][1] < rates[GO][1]  # a real safety/progress trade-off


# ---------------------------------------------------------------- 2. same actions and safety layer for everyone
def test_illegal_actions_are_refused_and_the_harness_falls_back():
    with pytest.raises(ValueError):
        world().step(7)

    class Bad(Policy):
        def decide(self, t):
            return 9

    row = run_episode(world(2), Bad(), 0.0)
    assert row["fallbacks"] == row["periods"] and row["actions"][GO] == row["periods"]


def test_safety_supervisor_is_policy_independent():
    """Same state → same safety decision, whatever system produced the state; replaying the same actions gives
    identical episodes (also proves common random numbers)."""
    rng = np.random.default_rng(0)
    actions = [int(a) for a in rng.integers(0, len(ACTIONS), 120)]
    a, b = world(3), world(3)
    oa, ob = drive(a, actions), drive(b, actions)
    assert oa == ob and a.log.interventions == b.log.interventions
    assert a.safety(0.8) == b.safety(0.8)


# ---------------------------------------------------------------- 3. APM failure leaves navigation and safety working
def test_apm_failure_falls_back_to_ordinary_navigation():
    predictor = ActionPredictor(WIDTH)
    predictor.weights = [(np.zeros((WIDTH + 4, 3), np.float32), np.zeros(3, np.float32))]

    class Crashing(B2):
        def observe(self, obs, t):
            if t > 5.0:
                raise RuntimeError("APM crashed")
            super().observe(obs, t)

    lt = Lifetime(seedmod.world_seed("E0", "P", 4), n_episodes=1)
    crashed = run_lifetime(lt, Crashing(predictor, 1.0, 0.5, 1.0, 0.25))
    ordinary = run_lifetime(lt, B0())
    assert crashed["episodes"][0]["policy_failed"] and crashed["fallbacks"] > 100
    assert crashed["goals_completed"] > 0
    assert abs(crashed["goals_completed"] - ordinary["goals_completed"]) <= 3


# ---------------------------------------------------------------- 4. standing still cannot succeed
def test_always_yield_does_not_succeed():
    class Freeze(Policy):
        def decide(self, t):
            return YIELD

    frozen = [run_lifetime(Lifetime(s, n_episodes=1), Freeze()) for s in seedmod.seeds("E0", "P", 8)]
    ordinary = [run_lifetime(Lifetime(s, n_episodes=1), B0()) for s in seedmod.seeds("E0", "P", 8)]
    assert np.mean([r["completion"] for r in frozen]) < 0.05
    assert np.mean([r["J"] for r in frozen]) < np.mean([r["J"] for r in ordinary])


# ---------------------------------------------------------------- 5. no future information, no hidden state
def test_observations_hold_no_hidden_state_or_future_information():
    allowed_obs = {"t", "v", "omega", "traction_margin", "energy", "goal_distance", "goal_bearing",
                   "goal_remaining_s", "goal_index", "entities", "events", "action"}
    allowed_seen = {"track", "kind", "identity", "identity_confidence", "distance", "bearing", "closing", "ttc_p"}
    assert {f.name for f in dataclasses.fields(Obs)} == allowed_obs
    assert {f.name for f in dataclasses.fields(Seen)} == allowed_seen
    a, b = world(5), world(5)
    oa = drive(a, [GO] * 10)
    ob = drive(b, [GO] * 10)
    for o in b.bodies:  # perturb only b's future: people's randomness and the remaining goals
        if o.rng is not None:
            o.rng = np.random.default_rng(999)
    b.goals[b._goal_i + 1:] = [(0.6, 0.6)] * (len(b.goals) - b._goal_i - 1)
    oa += drive(a, [GO] * 10)
    ob += drive(b, [GO] * 10)
    assert oa[:100] == ob[:100] and oa[100:] != ob[100:]
    assert all(o.t == pytest.approx((i + 1) * 0.05) for i, o in enumerate(oa))


def _reaches_world(obj, depth=0, seen=None) -> bool:
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 6:
        return False
    seen.add(id(obj))
    if isinstance(obj, WheeledCtrlWorld) or type(obj).__name__ == "_Body":
        return True
    if isinstance(obj, dict):
        return any(_reaches_world(v, depth + 1, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(_reaches_world(v, depth + 1, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return any(_reaches_world(v, depth + 1, seen) for v in vars(obj).values())
    return False


def test_decision_systems_hold_no_reference_to_the_simulator():
    predictor = ActionPredictor(WIDTH)
    predictor.weights = [(np.zeros((WIDTH + 4, 3), np.float32), np.zeros(3, np.float32))]
    for policy in (B0(), Explore(0), B2(predictor, 1.0, 0.5, 1.0, 0.25)):
        run_lifetime(Lifetime(seedmod.world_seed("E0", "P", 10), n_episodes=1), policy)
        assert not _reaches_world(policy)


# ---------------------------------------------------------------- 6. common random numbers
def test_goal_sequence_and_population_do_not_depend_on_the_policy():
    lt = Lifetime(seedmod.world_seed("E0", "P", 6), n_episodes=2)
    assert lt.population() == lt.population()
    w1, w2 = lt.world(0), lt.world(0)
    drive(w1, [GO] * 30)
    drive(w2, [YIELD] * 30)
    assert w1.goals == w2.goals
    issued1, issued2 = w1.goals[:w1._goal_i + 1], w2.goals[:w2._goal_i + 1]
    assert issued1[:min(len(issued1), len(issued2))] == issued2[:min(len(issued1), len(issued2))]
    assert [(o.oid, o.identity) for o in w1.bodies] == [(o.oid, o.identity) for o in w2.bodies]


# ---------------------------------------------------------------- 7. duplicate and missed events
def test_collision_events_match_ground_truth_contact_episodes():
    total = 0
    for i in range(10):
        w = world(20 + i)
        while not w.done:
            w.step(GO)
        expected, last = [], {}
        for t, track in w.log.contact_steps:
            if track not in last or t - last[track] > 1.0 + 1e-9:
                expected.append((round(t, 6), track))
            last[track] = t
        events = [(round(c[0], 6), c[1]) for c in w.log.collisions]
        assert events == expected  # no duplicates, no misses
        total += len(events)
    assert total > 0


def test_adapter_messages_validate_retract_stale_predictions_and_none_are_rejected():
    w = world(7)
    adapter = WheeledCtrlAdapter(Emitter("w"))
    apm = APM(profile=adapter.profile())
    apm.ingest(adapter.profile(), now_s=0.0)
    live, retracted, outcomes = set(), 0, 0
    for _ in range(1200):
        obs = w.step(GO)
        for m in adapter.step(obs, obs.t):
            validate(m)
            assert apm.ingest(m) is None
            if isinstance(m, OutcomeHypothesis):
                if m.probability == 0:
                    assert m.target in live
                    retracted += 1
                    live.discard(m.target)
                else:
                    live.add(m.target)
            outcomes += isinstance(m, OutcomeSignal) and m.kind == "collision"
        apm.step(obs.t)
        closing = {e.track for e in obs.entities if e.ttc_p > 0}
        held = {h.target for h in apm.context.hypotheses.values() if h.probability > 0}
        assert held <= closing  # D3 fixed: no stale non-zero prediction is held
    assert retracted > 0 and outcomes == len(w.log.collisions)


def test_explicit_memory_matches_the_apm_memory_on_the_same_stream():
    w = world(8)
    adapter = WheeledCtrlAdapter(Emitter("w"))
    apm = APM(profile=adapter.profile())
    memory = ObsMemory()
    for _ in range(1200):
        obs = w.step(GO)
        for m in adapter.step(obs, obs.t):
            apm.ingest(m)
        apm.step(obs.t)
        memory.observe(obs, obs.t)
    apm.step(w.t + 5)
    memory.expire(w.t + 5)
    a = {k: (r.encounters, r.counts.get("collision", 0.0)) for k, r in apm.entities.records.items()}
    b = {k: (r.encounters, r.counts.get("collision", 0.0)) for k, r in memory.mem.records.items()}
    assert a == b and a


# ---------------------------------------------------------------- 8. labels use only post-decision outcomes
def test_predictor_labels_are_strictly_after_the_decision():
    positives = 0
    for i in range(12):
        lt = Lifetime(seedmod.world_seed("E0", "P", 30 + i), n_episodes=1, p_careless=0.8)
        rec, pol = DecisionRecorder(), Explore(i)
        pol.begin_lifetime()
        w = lt.world(0)
        run_episode(w, pol, 0.0, rec)
        col_times = [c[0] for c in w.log.collisions]
        for r in rec.rows:
            assert r["y_col"] == float(any(r["t"] < c <= r["t"] + 2.0 + 1e-9 for c in col_times))
            assert r["t"] <= w.t - 2.0 + 1e-9  # censored windows at the episode end are dropped
            positives += r["y_col"]
    assert positives > 0


# ---------------------------------------------------------------- 9. frozen, disjoint partitions
def test_partitions_are_disjoint_and_never_reproduce_old_seeds():
    prints = {}
    for part in seedmod.PARTITIONS:
        for exp in seedmod.EXPERIMENTS:
            for s in seedmod.seeds(exp, part, 50):
                fp = seedmod.fingerprint(s)
                assert fp not in prints
                prints[fp] = (exp, part)
    old = {seedmod.fingerprint(np.random.SeedSequence(s)) for s in seedmod.V01_WHEELED_INT_SEEDS}
    assert not old & set(prints)
    assert child(seedmod.world_seed("E2", "X", 0), 1).spawn_key == (2, 4, 0, 1)


# ---------------------------------------------------------------- statistics helpers
def test_cluster_bootstrap_is_cluster_level_and_detects_a_real_difference():
    from apm_research.stats import holm, relative_loss_difference
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 50)
    base = rng.random((3, 2000))
    same = relative_loss_difference(base, base, groups, n_boot=500)
    assert same["relative_difference"] == pytest.approx(0.0) and same["ci95"][0] <= 0 <= same["ci95"][1]
    better = relative_loss_difference(base * 0.8, base, groups, n_boot=500)
    assert better["relative_difference"] == pytest.approx(-0.2) and better["ci95"][1] < 0
    # duplicating every sample inside its cluster must not shrink the interval (no step-level pseudo-replication)
    dup = relative_loss_difference(np.repeat(base * 0.95, 2, 1), np.repeat(base, 2, 1), np.repeat(groups, 2),
                                   n_boot=500)
    one = relative_loss_difference(base * 0.95, base, groups, n_boot=500)
    assert dup["se"] == pytest.approx(one["se"], rel=0.3)
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
