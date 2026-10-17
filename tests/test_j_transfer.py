"""Programme J: extended encounters, the decay-only control, the decision read-out and J1's adaptation metrics."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from apm_revise.belief import TypeModel
from apm_testbeds.wheeled_ctrl import GO, SLOW, YIELD
from apm_transfer.controls import DecayOnlyMemory
from apm_transfer.encounter import EncounterX

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "J_validation_transfer" / "scripts"))
import j1_metrics as M  # noqa: E402


def _model() -> TypeModel:
    E = np.full((2, 8), 0.01)
    E[0, 7], E[1, 2] = 0.93, 0.93
    E /= E.sum(1, keepdims=True)
    return TypeModel("dense", np.array([0.6, 0.4]), E, np.array([0.01, 0.03]))


def test_encounter_bands_and_decisions():
    e = EncounterX("t", 0.0)
    for _ in range(4):
        e.observe_x(1.0, 0.5, GO, math.radians(10), others_near=False)
    for _ in range(4):
        e.observe_x(1.0, 0.18, SLOW, math.radians(10), others_near=True)
    e.observe_x(1.2, 0.0, YIELD, 0.0, others_near=False)  # stopped: near and careful, not observable
    assert e.near_frames == 9 and e.careful_frames == 5
    assert e.action_band() == 0  # mean observable speed 0.34 m/s: slow band
    assert e.context_band() == 2  # frontal, crowded in exactly half of the frames (not more): not crowded
    assert abs(e.careful_fraction - 5 / 9) < 1e-12
    assert EncounterX("u", 0.0).action_band() is None


def test_decay_only_memory_stops_learning_but_logs():
    mem = DecayOnlyMemory(_model(), lam=0.0, freeze_after_s=10.0)
    person = SimpleNamespace(kind="person", track="p", identity="ped-0", identity_confidence=0.9, distance=0.07,
                             bearing=0.0, closing=0.3)
    for k in range(5):
        mem.observe(SimpleNamespace(t=k * 0.05, v=0.5, action=GO, entities=(person,), events=()), k * 0.05)
    mem.observe(SimpleNamespace(t=3.0, v=0.5, action=GO, entities=(), events=()), 3.0)
    before = mem.posterior("ped-0").copy()
    assert before[1] > 0.4  # learned before the freeze
    for k in range(5):
        mem.observe(SimpleNamespace(t=20 + k * 0.05, v=0.5, action=GO, entities=(person,), events=()), 20 + k * 0.05)
    mem.observe(SimpleNamespace(t=30.0, v=0.5, action=GO, entities=(), events=()), 30.0)
    assert np.allclose(mem.posterior("ped-0"), before)  # not learned after the freeze
    assert len(mem.log) == 2 and mem.log[-1][7] is None


def _lifetime(traj: list[float], to: str = "careful") -> dict:
    snaps = [{"ped-0": [p, 0.9], "ped-1": [0.01, 0.9]} for p in traj]
    truth = [{"ped-0": ("careless" if e < 15 else to), "ped-1": "careful"} for e in range(len(traj))]
    return {"roles": {"switched": ["ped-0"], "to": to}, "snapshots": snaps, "truth": truth, "encounters": []}


def test_decisive_adaptation_rejects_drift_to_the_prior():
    cal = {"mu_careless": 0.03, "mu_careful": 0.01, "threshold": 0.02, "gap": 0.02}
    revised = _lifetime([0.03] * 15 + [0.025, 0.012, 0.011] + [0.01] * 12)
    drifted = _lifetime([0.03] * 15 + [0.025, 0.0199, 0.0199] + [0.0195] * 12)  # just below θ: "adapts", not deep
    assert M._items(revised, cal, decisive=False) == [(2.0, True)]
    assert M._items(revised, cal, decisive=True) == [(2.0, True)]
    assert M._items(drifted, cal, decisive=False) == [(2.0, True)]
    assert M._items(drifted, cal, decisive=True) == [(15.0, False)]  # never beyond the midpoint of θ and μ_careful
    blip = _lifetime([0.03] * 15 + [0.012, 0.03] + [0.03] * 13)  # one-episode excursion is not decisive
    assert M._items(blip, cal, decisive=True) == [(15.0, False)]


def test_readout_batches_match_single_calls():
    from apm_transfer.readout import DecisionReadout

    class Pred:  # collision risk rises with the memory input; SLOW/YIELD safer but slower
        def predict(self, X, A):
            p = X[:, 13] * (1.0 - 0.5 * (A > 0))
            return p, np.zeros(len(A)), 1.0 - 0.1 * A
    bank = np.zeros((5, 41), np.float32)
    ro = DecisionReadout(Pred(), lam_c=20.0, lam_s=0.0, bank=bank, multipliers=np.ones(5))
    shares = ro.careful_shares([(0.0, 0.5), (0.9, 0.5)])
    assert shares == [ro.careful_share(0.0, 0.5), ro.careful_share(0.9, 0.5)]
    assert shares[0] == 0.0 and shares[1] == 1.0


def test_utility_mapping_needs_a_proximity_value_to_depend_on_belief():
    from apm_transfer.expression_map import GRID, TimeModel, utility_intent
    tm = TimeModel([20.0, 10.0, -17.0, -1.5, 6.5])  # nearer is slower, faster tempo is quicker (as fitted in J6-T)
    ps = np.array([[0.13, 0.11, 0.11], [0.61, 0.45, 0.23]])  # P(strong | type, band); band 0 = nearest
    picks = {utility_intent(np.array([1 - b, b]), tm, ps, 10.0)[0] for b in (0.0, 0.5, 1.0)}
    assert len(picks) == 1 and GRID[picks.pop()].proximity == 0.0  # no value: always far, for every belief
    near = [GRID[utility_intent(np.array([1 - b, b]), tm, ps, 40.0, 5.0)[0]].proximity for b in (0.0, 1.0)]
    assert near[0] > near[1]  # with a proximity value, a riskier belief stops further away
