"""Programme H internal-state models: experience dependence, controls and bounded expression (no physics)."""
from __future__ import annotations

import numpy as np

from apm_express.develop import HISTORIES, kind_of, schedule
from apm_express.states import (
    CAUTIOUS,
    RELAXED,
    Constant,
    EntityMemory,
    FixedStyle,
    Obs,
    PADState,
    Probe,
    ScalarEMA,
    SmallState,
    caution,
    intent_of,
)

B1 = Probe("B1", "B", "plain")


def _stream(history: str, seed: int) -> list[Obs]:
    rng = np.random.default_rng(seed)
    return [Obs(i, kind_of(i), z, float(j), 0.8 if j else 0.1) for i, z, j in schedule(history, rng)]


def _after(model, history, seed=0):
    model.reset()
    for ob in _stream(history, seed):
        model.observe(ob)
    return model


def test_no_state_and_fixed_style_ignore_history():
    for m in (Constant(), FixedStyle()):
        a = intent_of(_after(m, "risky_B"), B1)
        b = intent_of(_after(m, "safe"), B1)
        assert a == b


def test_entity_memory_is_history_and_identity_specific():
    m = EntityMemory()
    risky = caution(_after(m, "risky_B").readout(B1))
    safe = caution(_after(EntityMemory(), "safe").readout(B1))
    assert risky > safe + 0.3
    m = _after(EntityMemory(), "risky_B")
    assert caution(m.readout(B1)) > caution(m.readout(Probe("A1", "A", "plain"))) + 0.3
    assert caution(m.readout(Probe("B3", "B", "plain"))) > caution(m.readout(Probe("A3", "A", "plain")))  # kind pool


def test_context_memory_separates_zones():
    m = _after(EntityMemory(context=True), "zone_B")
    assert caution(m.readout(Probe("B1", "B", "marked"))) > caution(m.readout(B1)) + 0.2


def test_global_states_respond_but_are_not_identity_specific():
    for cls in (ScalarEMA, SmallState, PADState):
        m = _after(cls(), "unpredictable")
        r_b, r_a = m.readout(B1), m.readout(Probe("A1", "A", "plain"))
        assert abs(r_b.p - r_a.p) < 1e-12
        assert m.readout(B1).p > _after(cls(), "safe").readout(B1).p


def test_expression_is_bounded_and_between_relaxed_and_cautious():
    for h in HISTORIES:
        for m in (EntityMemory(), PADState(), SmallState()):
            it = intent_of(_after(m, h), B1).sanitized()
            assert min(RELAXED.tempo, CAUTIOUS.tempo) - 1e-9 <= it.tempo <= max(RELAXED.tempo, CAUTIOUS.tempo) + 1e-9
            assert 0.0 <= it.hesitation <= 1.0 and 0.0 <= it.scanning <= 1.0
