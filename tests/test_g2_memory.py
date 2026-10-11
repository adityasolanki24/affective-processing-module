"""G2 belief-memory tests: M0 with all-encounter evidence equals programme F's memory; exposure-gated evidence
never learns from encounters without exposure; revision models move faster than cumulative counts after a change."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from apm_select.memory import BeliefMemory, ChangePoint, Cumulative, HiddenState, Window

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "E2" / "predictor.json"


@pytest.fixture(scope="module")
def gc():
    sys.path.insert(0, str(ROOT / "experiments" / "G_architecture_selection" / "scripts"))
    import g_common
    g_common.init(str(PRED))
    return g_common


@pytest.mark.skipif(not PRED.exists(), reason="frozen predictor not present")
def test_m0_all_evidence_equals_programme_f_memory(gc):
    base = {"kind": "regulated", "regulator": "static", "lam_c": 1.0, "lam_s": 2.0}
    for idx in (0, 1):
        ref = gc.run_job({"spec": base, "experiment": "G2", "partition": "P", "index": idx, "n_episodes": 3})
        new = gc.run_job({"spec": {**base, "memory": {"model": "M0", "evidence": "all"}}, "experiment": "G2",
                          "partition": "P", "index": idx, "n_episodes": 3})
        assert new["actions"] == ref["actions"] and new["totals"]["collisions"] == ref["totals"]["collisions"]


def _seq(beliefs, ys, prior=0.1):
    out = []
    for y in ys:
        for b in beliefs:
            b.update(y, 1.0, prior)
        out.append([b.p(prior)[0] for b in beliefs])
    return np.array(out)


def test_revision_models_adapt_faster_than_cumulative():
    rng = np.random.default_rng(0)
    ys = list((rng.random(30) < 0.3).astype(float)) + list((rng.random(30) < 0.02).astype(float))
    bs = [Cumulative(), Cumulative(gamma=0.85), Window(10), ChangePoint(0.05), HiddenState(0.02, 0.3, 0.05)]
    p = _seq(bs, ys)
    after = p[-1]
    assert all(after[i] < after[0] for i in range(1, len(bs))), after
    assert np.all((p >= 0) & (p <= 1))


def test_changepoint_run_length_is_a_distribution():
    b = ChangePoint(0.1)
    for y in [1, 0, 0, 1, 0] * 30:
        b.update(float(y), 0.9, 0.1)
        assert b.r.sum() == pytest.approx(1.0) and len(b.r) <= ChangePoint.RMAX


class _E:
    def __init__(self, track, d, ident="ped-0", conf=0.9):
        self.track, self.kind, self.identity, self.identity_confidence, self.distance = track, "person", ident, conf, d


class _O:
    def __init__(self, ents, v=0.5, action=0, events=()):
        self.entities, self.v, self.action, self.events = ents, v, action, events


def test_exposed_evidence_ignores_unexposed_and_avoided_encounters():
    mem_all, mem_exp = BeliefMemory("M0", {}, "all"), BeliefMemory("M0", {}, "exposed")
    for m in (mem_all, mem_exp):
        t = 0.0
        for k in range(3):  # three far-away encounters, then one avoided close encounter while yielding
            m.observe(_O([_E(f"a{k}", 2.0)]), t)
            t += 0.05
            m.observe(_O([]), t + 2.0)
            t += 2.1
        m.observe(_O([_E("b", 0.3)], v=0.0, action=2), t)
        m.observe(_O([]), t + 2.0)
    assert mem_all.beliefs["ped-0"].n == pytest.approx(4 * 0.9)
    assert "ped-0" not in mem_exp.beliefs and mem_exp.pool.n == 0
    assert mem_exp.counts["unexposed"] == 3 and mem_exp.counts["avoided"] == 1


def test_exposed_encounter_and_collision_are_evidence():
    m = BeliefMemory("M1", {"gamma": 0.9}, "exposed")
    m.observe(_O([_E("a", 0.3)], v=0.5), 0.0)
    m.observe(_O([]), 2.0)
    m.observe(_O([_E("b", 0.1)], v=0.5, events=(("collision", "b", False),)), 3.0)
    m.observe(_O([]), 5.0)
    b = m.beliefs["ped-0"]
    assert b.n == pytest.approx(0.9 * 0.9 + 0.9) and b.k == pytest.approx(0.9)
    assert m.counts["exposed"] == 1 and m.counts["collision"] == 1


def test_m6_numpy_inference_matches_torch_gru():
    import importlib.util
    torch = pytest.importorskip("torch")
    from apm_select.memory import Neural
    path = ROOT / "experiments" / "G_architecture_selection" / "scripts" / "g2_m6_train.py"
    spec = importlib.util.spec_from_file_location("g2_m6_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(3)
    model = mod.M6()
    belief = Neural(mod.export(model))
    h = torch.zeros(mod.HIDDEN)
    prior = 0.12
    for y, w in [(0.0, 0.9), (1.0, 0.9), (0.0, 0.4), (0.0, 0.9), (1.0, 1.0)]:
        with torch.no_grad():
            logit = float(model.out(h)[0]) + np.log(prior / (1 - prior))
        assert belief.p(prior)[0] == pytest.approx(1 / (1 + np.exp(-logit)), abs=1e-6)
        with torch.no_grad():
            h = model.cell(torch.tensor([[y, w]], dtype=torch.float32), h[None])[0]
        belief.update(y, w, prior)


@pytest.mark.skipif(not PRED.exists(), reason="frozen predictor not present")
def test_probing_only_replaces_avoidance_with_slow_within_budget(gc):
    base = {"kind": "regulated", "regulator": "static", "lam_c": 2.5, "lam_s": 1.5}
    mem = {"model": "M1", "params": {"gamma": 0.9}, "evidence": "exposed"}
    ref = gc.run_job({"spec": {**base, "memory": mem}, "experiment": "G2", "partition": "P", "index": 3,
                      "n_episodes": 2})
    pol = gc.make_policy({**base, "memory": {**mem, "probe": {"mode": "random", "budget": 3, "p_random": 1.0}}})
    from apm_research.harness import run_lifetime
    r = run_lifetime(gc.lifetime_for("G2", "P", 3, 2), pol)
    assert 0 < pol.probes <= 3 * 2 and len(pol.probe_log) == pol.probes
    from apm_testbeds.wheeled_ctrl import DETOUR, SLOW, YIELD
    assert all(rule in (YIELD, DETOUR) and done == SLOW and k <= 3 for k, rule, done in pol.probe_log)
    assert sum(r["actions"]) == sum(ref["actions"])


def test_time_decay_reverts_toward_prior_without_new_evidence():
    from apm_select.memory import TimeDecay
    now = [0.0]
    b = TimeDecay(100.0, lambda: now[0])
    for _ in range(10):
        b.update(1.0, 1.0, 0.05)
    high = b.p(0.05)[0]
    now[0] = 1000.0  # no evidence for ten time constants
    low = b.p(0.05)[0]
    assert high > 0.8 and low < 0.1 and b.n < 0.01
