"""Programme J2: stress conditions keep the frozen testbed intact; conditioned evidence models and the observer."""
from __future__ import annotations

import numpy as np

import apm_testbeds.wheeled_ctrl as W
from apm_research.harness import run_episode
from apm_testbeds.wheeled_ctrl import GO
from apm_transfer.conditioned import BANDS, category_x, fit_conditional
from apm_transfer.encounter import EncounterX
from apm_transfer.stress import CONDITIONS, make_stress


class Go:
    def begin_episode(self, t0): ...
    def observe(self, obs, t): ...
    def decide(self, t): return GO
    def end_episode(self, t): ...


def test_every_condition_builds_and_runs_one_episode():
    for cond in CONDITIONS:
        lt, roles, wrap, ctx = make_stress(cond, "S1", np.random.SeedSequence(1), episodes=2)
        world = lt.world(0)
        before = {o.oid: o.identity for o in world.bodies if o.kind == "person"}
        pol = wrap(Go())
        if ctx:
            with ctx():
                run_episode(world, pol, 0.0)
        else:
            run_episode(world, pol, 0.0)
        after = {o.oid: o.identity for o in world.bodies if o.kind == "person"}
        assert before == after, cond  # acting dispositions never leak into identities
        assert roles["condition"] == cond
    assert W.V_MAX == 0.8 and W.MODES[GO][0] == 0.8  # physics patches are undone


def test_erratic_and_drift_roles():
    lt, roles, _, _ = make_stress("erratic", "S3", np.random.SeedSequence(2))
    assert 1 <= len(roles["erratic"]) <= 4
    lt, roles, _, _ = make_stress("drift", "S1", np.random.SeedSequence(3))
    w = lt.world(12)
    assert set(w.drift_ids) == {"ped-0", "ped-1"} and w.drift_ids["ped-0"] == ("careless", "careful")


def test_conditioned_category_and_fit():
    e = EncounterX("t", 0.0)
    for _ in range(5):
        e.observe_x(0.12, 0.6, GO, 0.0, others_near=False)
    assert category_x(e, "dense") == 3
    assert category_x(e, "dense_action") == 1 * 8 + 3     # fast band
    assert category_x(e, "dense_context") == 2 * 8 + 3    # frontal, not crowded
    rng = np.random.default_rng(0)
    n, B, C = 300, BANDS["dense_action"], 8
    z = rng.random(n) < 0.4
    counts = np.zeros((n, B, C))
    for i in range(n):
        for b in range(B):  # the slow band is uninformative; the fast band separates the types
            p = np.full(C, 1 / C) if b == 0 else (np.r_[0.02, 0.6, np.full(6, 0.38 / 6)] if z[i]
                                                   else np.r_[0.01, np.full(6, 0.04), 0.75])
            counts[i, b] = rng.multinomial(6, p / p.sum())
    m = fit_conditional(counts, np.full(n, 12.0), counts[:, :, 0].sum(1), 2, "dense_action", restarts=4)
    E = m.E.reshape(2, B, C)
    assert np.allclose(E.sum(2), 1.0)
    assert np.abs(E[:, 0] - 1 / C).max() < 0.08            # slow band: both types ≈ uniform
    assert abs(E[0, 1, 7] - E[1, 1, 7]) > 0.4               # fast band: the types differ
