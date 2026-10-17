"""Programme J4: NERVA type-belief memory behind H's interface, history schedules, restart persistence."""
from __future__ import annotations

import json

import numpy as np

from apm_express.states import Probe, lookup_intent
from apm_revise.belief import TypeModel
from apm_transfer.nerva_memory import HISTORIES, ObsX, TypeStateModel, entity_type, lifetime_types


def _model() -> TypeModel:
    E = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    return TypeModel("acc3", np.array([0.6, 0.4]), E, np.array([0.05, 0.6]))


def test_type_memory_learns_pools_forgets_and_restores():
    m = TypeStateModel(_model(), lam=0.0, kind_prior=True)
    prior_p = m.readout(Probe("B1", "B", "plain")).p
    for _ in range(4):
        m.observe(ObsX("B1", "B", "plain", 1.0, 1.0, category=2))
    hi = m.readout(Probe("B1", "B", "plain"))
    assert hi.p > prior_p + 0.3 and hi.u < 0.3
    novel = m.readout(Probe("B3", "B", "plain")).p  # kind pooling: a new B starts above the plain prior
    assert prior_p < novel < hi.p
    other = m.readout(Probe("A1", "A", "plain")).p
    assert abs(other - prior_p) < 1e-12
    restored = TypeStateModel(_model(), lam=0.0, kind_prior=True)
    restored.load_state(json.loads(json.dumps(m.state())))
    assert restored.readout(Probe("B1", "B", "plain")) == hi
    assert lookup_intent(restored.readout(Probe("B1", "B", "plain"))) == lookup_intent(hi)
    f = TypeStateModel(_model(), lam=0.5)
    f.observe(ObsX("B1", "B", "plain", 1.0, 1.0, category=2))
    early = f.readout(Probe("B1", "B", "plain")).p
    for _ in range(20):
        f.observe(ObsX("C1", "C", "plain", 0.0, 0.0, category=0))
    assert abs(f.readout(Probe("B1", "B", "plain")).p - float(_model().pi @ _model().r)) < abs(early - 0.3)


def test_histories():
    rng = np.random.default_rng(0)
    for h in HISTORIES:
        assert len(lifetime_types(h, rng)) == 48
    assert entity_type("change_B", "B1", "plain", 3, rng) == "pushy"
    assert entity_type("change_B", "B1", "plain", 30, rng) == "calm"
    assert entity_type("late_B", "B1", "plain", 30, rng) == "pushy"
    assert entity_type("zone_B", "A1", "marked", 3, rng) == "pushy"
    assert entity_type("zone_B", "B1", "plain", 3, rng) == "calm"
