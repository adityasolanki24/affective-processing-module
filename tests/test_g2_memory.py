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
