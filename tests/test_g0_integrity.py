"""Programme G0 integrity tests: equivalence to programme-F systems, no behavioural effect of instrumentation,
causality (no future information), no hidden state in observations, seed independence, equal tuning budgets,
degenerate-policy checks on the utility, and negative-control replay."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "experiments" / "G_architecture_selection"
PRED = ROOT / "runs" / "E2" / "predictor.json"
pytestmark = pytest.mark.skipif(not PRED.exists(), reason="frozen predictor not present")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gc():
    sys.path.insert(0, str(G / "scripts"))
    mod = _load("g_common", G / "scripts" / "g_common.py")
    mod.init(str(PRED))
    return mod


def _run(gc, spec, episodes=2, condition="nominal", index=0, partition="P"):
    return gc.run_job({"spec": spec, "experiment": "G1", "partition": partition, "index": index,
                       "n_episodes": episodes, "condition": condition, "keep_readouts": True})


def test_static_regulator_equals_b1(gc):
    from apm_research.harness import run_lifetime
    from apm_research.policies import B1
    lt = gc.lifetime_for("G1", "P", 0, 2)
    ref = run_lifetime(lt, B1(gc._STATE["predictor"], 1.0, 2.0))
    r = _run(gc, {"kind": "regulated", "regulator": "static", "lam_c": 1.0, "lam_s": 2.0})
    assert r["actions"] == ref["actions"] and r["totals"]["collisions"] == ref["collisions"]
    assert r["totals"]["goals_completed"] == ref["goals_completed"]


def test_r6_pad_equals_programme_f_b2(gc):
    from apm_research.harness import run_lifetime
    from apm_research.policies import B2
    lt = gc.lifetime_for("G1", "P", 1, 2)
    ref = run_lifetime(lt, B2(gc._STATE["predictor"], 0.5, 2.0, 2.0, 0.25))
    r = _run(gc, {"kind": "b2v", "mode": "pad", "lam_c": 0.5, "lam_s": 2.0, "k": 2.0, "k_t": 0.25}, index=1)
    assert r["actions"] == ref["actions"] and r["totals"]["collisions"] == ref["collisions"]
    assert r["totals"]["distance"] == pytest.approx(ref["distance"])


def test_instrumentation_does_not_change_behaviour(gc):
    from apm_research.harness import run_lifetime
    from apm_select.instrument import Instrumented
    spec = {"kind": "regulated", "regulator": "ema", "lam_c": 1.0, "lam_s": 2.0,
            "params": {"tau": 5.0, "k": 2.0, "c": 0.5}}
    lt = gc.lifetime_for("G1", "P", 2, 2)
    plain = run_lifetime(lt, gc.make_policy(spec))
    wrapped_policy = Instrumented(gc.make_policy(spec))
    wrapped = run_lifetime(lt, wrapped_policy)
    assert plain["actions"] == wrapped["actions"] and plain["collisions"] == wrapped["collisions"]
    s = wrapped_policy.stats
    assert s["decisions"] == sum(wrapped["actions"]) and s["person_encounters"] >= s["close_encounters"] >= 0


@pytest.mark.parametrize("family", ["R2", "R3", "R4"])
def test_regulators_are_causal(gc, family):
    """m at a decision depends only on observations up to that decision: feeding a prefix of the stream gives the
    same multiplier as the full run had at that point."""
    from apm_select.regulators import make_regulator
    from apm_select.tuning import family_configs
    spec = family_configs(family, 4, 0)[1]
    lt = gc.lifetime_for("G1", "P", 3, 1)
    world = lt.world(0)
    stream = [world.observe_now()]
    while not world.done and len(stream) < 400:
        stream.append(world.step(0))
    full = make_regulator(spec["regulator"], spec["params"])
    full.reset()
    ms = []
    for o in stream:
        full.update(o, o.t)
        ms.append(full.multiplier())
    for cut in (50, 200, 399):
        pre = make_regulator(spec["regulator"], spec["params"])
        pre.reset()
        for o in stream[:cut + 1]:
            pre.update(o, o.t)
        assert pre.multiplier() == pytest.approx(ms[cut])


def test_observations_expose_no_hidden_state():
    from apm_testbeds.wheeled_ctrl import Obs, Seen
    fields = {f.name for f in dataclasses.fields(Obs)} | {f.name for f in dataclasses.fields(Seen)}
    for hidden in ("disposition", "mu", "friction", "careless", "truth", "x", "y", "theta"):
        assert hidden not in fields


def test_regulators_need_only_observation_fields(gc):
    """Run a regulated policy on observations stripped to a plain namespace of Obs fields (no simulator access)."""
    from types import SimpleNamespace

    from apm_select.regulators import EMA, BayesHazard, Hysteresis, LearnedScalar
    lt = gc.lifetime_for("G1", "P", 4, 1)
    world = lt.world(0)
    regs = [EMA(3.0, 1.0, 0.5), Hysteresis(1.0, 20.0, 1.0, 0.5), BayesHazard(60.0, 1.0, 1.0),
            LearnedScalar(np.ones(8) * 0.1)]
    for r in regs:
        r.reset()
    for _ in range(100):
        o = world.step(0)
        plain = SimpleNamespace(**{f.name: getattr(o, f.name) for f in dataclasses.fields(o)})
        for r in regs:
            r.update(plain, o.t)
            assert np.isfinite(r.multiplier()) and r.multiplier() >= 0


def test_g_seeds_independent_of_f():
    sg = _load("g_seeds", G / "seeds.py")
    sf = _load("f_seeds", ROOT / "experiments" / "F_behavioral_learning" / "seeds.py")
    g = {sg.fingerprint(sg.world_seed(e, p, i)) for e in ("G0", "G1", "G2") for p in sg.PARTITIONS for i in range(50)}
    f = {sf.fingerprint(sf.world_seed(e, p, i)) for e in sf.EXPERIMENTS for p in sf.PARTITIONS for i in range(50)}
    assert len(g) == 3 * len(sg.PARTITIONS) * 50 and not g & f


def test_equal_tuning_budgets():
    from apm_select.tuning import FAMILIES, family_configs
    sizes = {f: len(family_configs(f, 48, 0)) for f in FAMILIES}
    assert set(sizes.values()) == {48}
    assert family_configs("R2", 48, 0) == family_configs("R2", 48, 0)  # deterministic
    for f in FAMILIES:
        for spec in family_configs(f, 48, 0):
            for name, (_kind, lo, hi) in FAMILIES[f]["space"].items():
                v = spec.get(name, spec.get("params", {}).get(name))
                assert v is not None and lo - 1e-6 <= v <= hi + 1e-6


def test_degenerate_policies_do_not_win_the_utility(gc):
    """Perpetual yielding and perpetual GO must score below B1 on the tuning utility J (no reward exploitation)."""
    from apm_research.harness import run_lifetime
    from apm_research.policies import B0, B1, Policy
    from apm_testbeds.wheeled_ctrl import YIELD

    class AlwaysYield(Policy):
        def decide(self, t):
            return YIELD
    lt = gc.lifetime_for("G1", "P", 5, 2)
    j = {name: run_lifetime(lt, p)["J"] for name, p in
         (("yield", AlwaysYield()), ("go", B0()), ("b1", B1(gc._STATE["predictor"], 1.0, 2.0)))}
    assert j["yield"] < j["b1"] and j["go"] < j["b1"]
    # A parked robot can still "complete" a goal when a later pre-generated goal lands within the 0.3 m tolerance
    # of where it stands (G0 audit: 4 of 237 goals over 20 pilot lifetimes); it never moves.
    y = run_lifetime(lt, AlwaysYield())
    assert y["distance"] == 0.0 and y["goals_completed"] <= 2 and j["yield"] < 0.25 * j["b1"]


def test_trace_replay_of_own_history_reproduces_decisions(gc):
    spec = {"kind": "regulated", "regulator": "ema", "lam_c": 1.0, "lam_s": 2.0,
            "params": {"tau": 5.0, "k": 2.0, "c": 0.5}}
    live = _run(gc, spec, index=6)
    replay = _run(gc, {**spec, "trace": live["readouts"]}, index=6)
    assert replay["actions"] == live["actions"] and replay["totals"]["collisions"] == live["totals"]["collisions"]


def test_miscalibration_condition_scales_estimates(gc):
    from apm_select.regulators import threat
    seen = []

    class Probe:
        name = "probe"
        def begin_lifetime(self): ...
        def begin_episode(self, t0): ...
        def observe(self, obs, t):
            seen.append(obs)
        def decide(self, t):
            return 0
        def end_episode(self, t): ...
    from apm_research.harness import run_lifetime
    lt = gc.lifetime_for("G1", "P", 7, 1)
    base = Probe()
    run_lifetime(lt, base)
    plain = list(seen)
    seen.clear()
    run_lifetime(lt, gc.MiscalibratedInput(Probe()))
    assert len(plain) == len(seen)
    for a, b in zip(plain, seen, strict=True):
        for ea, eb in zip(a.entities, b.entities, strict=True):
            assert eb.ttc_p == pytest.approx(min(1.0, 3 * ea.ttc_p))
    assert max(threat(o) for o in seen) >= max(threat(o) for o in plain)


def test_condition_worlds_differ_as_specified(gc):
    dense = gc.lifetime_for("G1", "P", 0, 1, "dense")
    sparse = gc.lifetime_for("G1", "P", 0, 1, "sparse")
    assert dense.n_people == 8 and dense.config.people == (4, 6) and sparse.config.people == (1, 2)
    n_dense = sum(b.kind == "person" for b in dense.world(0).bodies)
    n_sparse = sum(b.kind == "person" for b in sparse.world(0).bodies)
    assert 4 <= n_dense <= 6 and 1 <= n_sparse <= 2
    json.dumps(gc.CONDITIONS)
