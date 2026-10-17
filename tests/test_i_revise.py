"""Programme I: evidence coding, type-model EM, change-aware beliefs, memory interface and value of information."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from apm_revise.belief import TypeBelief, TypeModel, fit_types, readout
from apm_revise.evidence import MIN_FRAMES, Encounter
from apm_revise.memory import RevisionMemory
from apm_revise.policy import bayes_risk, expected_posterior_risk
from apm_testbeds.wheeled_ctrl import GO, YIELD


def _model(coding: str = "dense") -> TypeModel:
    C = 8 if coding == "dense" else 2
    E = np.full((2, C), 1.0 / C)
    E[0] = np.r_[0.01, np.full(C - 2, 0.01), 1.0][:C]
    E[1] = np.r_[0.05, np.full(C - 1, 0.95 / (C - 1))][:C]
    E /= E.sum(1, keepdims=True)
    return TypeModel(coding, np.array([0.7, 0.3]), E, np.array([0.01, 0.03]))


def test_encounter_categories_and_censoring():
    e = Encounter("t", 0.0)
    for _ in range(MIN_FRAMES - 1):
        e.observe(0.12, 0.5, GO)
    assert e.category("dense") is None  # too few observable frames: censored
    e.observe(0.12, 0.5, GO)
    assert e.category("dense") == 1 + 2   # 0.10 <= 0.12 < 0.15
    assert e.category("sparse") == 1
    stopped = Encounter("s", 0.0)
    for _ in range(20):
        stopped.observe(0.05, 0.0, YIELD)  # the robot is stopped: people's behaviour is not observable
    assert stopped.category("dense") is None and stopped.avoided
    stopped.collision = True
    assert stopped.category("dense") == 0 and stopped.category("sparse") == 0


def test_em_recovers_two_types_without_labels():
    rng = np.random.default_rng(1)
    true = _model()
    z = rng.random(400) < 0.3
    counts = np.stack([rng.multinomial(6, true.E[int(k)]) for k in z]).astype(float)
    enc = np.full(400, 8.0)
    coll = counts[:, 0]
    m = fit_types(counts, enc, coll, 2, "dense", restarts=5, seed=0)
    assert abs(m.pi[1] - 0.3) < 0.07
    assert np.abs(m.E - true.E).max() < 0.08
    assert m.r[0] < m.r[1]
    assert fit_types(counts, enc, coll, 1, "dense").bic() > m.bic()  # BIC prefers the true K


def test_belief_relaxes_to_prior_and_updates():
    m = _model()
    b = TypeBelief(m, lam=0.01, t=0.0)
    b.update(0, 1.0, 0.0)  # collision: evidence for the risky type
    hi = b.at(0.0)[1]
    assert hi > m.pi[1]
    later = b.at(100.0)[1]
    assert abs(later - (m.pi[1] + (hi - m.pi[1]) * math.exp(-1.0))) < 1e-12
    static = TypeBelief(m, lam=0.0, t=0.0)
    static.update(7, 1.0, 0.0)
    assert static.at(1e6)[0] == static.at(0.0)[0]
    p, conf = readout(m, np.array([0.5, 0.5]))
    assert abs(p - 0.02) < 1e-12 and abs(conf) < 1e-12


def test_memory_interface_and_thompson():
    m = _model()
    mem = RevisionMemory(m, lam=0.0)
    person = SimpleNamespace(kind="person", track="p1", identity="ped-0", identity_confidence=0.9, distance=0.08,
                             bearing=0.0, closing=0.3)
    for k in range(5):
        mem.observe(SimpleNamespace(t=k * 0.05, v=0.5, action=GO, entities=(person,), events=()), k * 0.05)
    mem.observe(SimpleNamespace(t=3.0, v=0.5, action=GO, entities=(), events=()), 3.0)
    assert mem.log[-1][3] == "exposed" and mem.log[-1][6] == 2
    assert mem.posterior_p("ped-0")[0] > readout(m, m.pi)[0]
    assert mem.p_collision("nobody", "person") == mem.posterior_p(None)
    ts = RevisionMemory(m, thompson=True, seed=3)
    ts.observe(SimpleNamespace(t=0.0, v=0.5, action=GO, entities=(person,), events=()), 0.0)
    assert ts.p_collision("ped-0", "person")[0] in (0.01, 0.03)
    assert ts.p_collision("ped-0", "person") == ts.p_collision("ped-0", "person")  # one draw per encounter


def test_value_of_information():
    m = _model()
    sure = np.array([1.0, 0.0])
    assert bayes_risk(sure) == 0.0 and expected_posterior_risk(sure, m.E) == 0.0
    unsure = np.array([0.5, 0.5])
    assert expected_posterior_risk(unsure, m.E) < bayes_risk(unsure)  # an informative encounter reduces risk
    flat = np.full((2, 8), 1 / 8)
    assert abs(expected_posterior_risk(unsure, flat) - bayes_risk(unsure)) < 1e-12  # uninformative: no value
