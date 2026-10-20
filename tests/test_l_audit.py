"""Programme L audit fixes: confirmed event times (A1), the risk cap (A3), camera field of view (A5), the change
model (A6)."""
from __future__ import annotations

from apm_transfer.recovery_metrics import confirmed_onset


def test_a1_lone_favourable_last_meeting_is_censored_not_an_event():
    assert confirmed_onset([False, False, True]) == (3.0, False)
    assert confirmed_onset([False, False, True], legacy=True) == (3.0, True)  # K's behaviour (the defect)


def test_a1_two_consecutive_favourable_observations_are_an_event_at_the_first():
    assert confirmed_onset([False, True, True]) == (2.0, True)
    assert confirmed_onset([True, True]) == (1.0, True)
    assert confirmed_onset([True, False, True, True, False]) == (3.0, True)


def test_a1_edge_cases():
    assert confirmed_onset([]) == (1.0, False)
    assert confirmed_onset([True]) == (1.0, False)
    assert confirmed_onset([True], legacy=True) == (1.0, True)
    assert confirmed_onset([True, False]) == (2.0, False)
    assert confirmed_onset([False, True, True, True], need=3) == (2.0, True)
    assert confirmed_onset([False, True, True], need=3) == (3.0, False)


def _all_risky_model():
    import numpy as np

    from apm_revise.belief import TypeModel
    # two types; the pushy type perturbs strongly at every distance band (strong = categories 6, 7)
    calm = [0.4, 0.3, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01]
    push = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.4, 0.48]
    cam_c, cam_p = [0.8, 0.12, 0.05, 0.0, 0.02, 0.0, 0.0, 0.01], [0.25, 0.05, 0.15, 0.2, 0.05, 0.1, 0.1, 0.1]
    E = np.array([calm * 3 + cam_c, push * 3 + cam_p], float) + 1e-3
    E = (E.reshape(2, 4, 8) / E.reshape(2, 4, 8).sum(2, keepdims=True)).reshape(2, 32)
    return TypeModel("adversarial", np.array([0.5, 0.5]), E, np.array([0.1, 0.9]))


def test_a3_no_safe_action_is_explicit_and_never_an_over_cap_approach():
    import numpy as np

    from apm_transfer.expression_map import GRID, TimeModel
    from apm_transfer.infoseek import HOLD, NO_SAFE_ACTION, Emissions, Task, action_cost, best_greedy, choose, cost
    m = _all_risky_model()
    tm = TimeModel([20.0, 10.0, -17.0, -1.5, 6.5])
    em = Emissions.from_model(m.E, camera_block=True)
    task = Task("handoff", d_task=0.3, retry_s=12.6)
    b = np.array([0.0, 1.0])  # certainly the pushy type: every grid approach exceeds the cap
    assert all(cost(i, 1, b, tm, em, task)[1] > task.risk_cap for i in range(len(GRID)))
    assert best_greedy(b, tm, em, task)[0] == NO_SAFE_ACTION
    legacy_i = best_greedy(b, tm, em, task, legacy=True)[0]
    assert legacy_i >= 0 and cost(legacy_i, 1, b, tm, em, task)[1] > task.risk_cap  # K's silent violation, reproduced
    for policy in ("greedy", "oracle", "uncertain", "voi"):
        for e in (em, Emissions(em.imu, None)):
            i, w, d = choose(policy, b, m.pi, tm, e, task, rng=np.random.default_rng(0))
            assert i < 0, (policy, i)  # never a grid approach
            assert action_cost(i, w, b, tm, e, task)[1] == 0.0  # predicted risk of the executed action is 0
            if i == HOLD:
                assert d["no_safe_action"] and d["standoff"] > 0


def test_a3_uncertainty_triggered_windows_respect_the_cap_per_encounter():
    import numpy as np

    from apm_transfer.expression_map import TimeModel
    from apm_transfer.infoseek import Emissions, Task, choose, cost
    m = _all_risky_model()
    tm = TimeModel([20.0, 10.0, -17.0, -1.5, 6.5])
    em = Emissions(Emissions.from_model(m.E, camera_block=True).imu, None)
    task = Task("handoff", d_task=0.3, retry_s=12.6)
    found = False
    for p in np.linspace(0.02, 0.6, 59):  # uncertain beliefs: one window within the cap, three beyond it
        b = np.array([1 - p, p])
        i, w, d = choose("uncertain", b, m.pi, tm, em, task, u_trigger=0.1)
        if i >= 0:
            assert cost(i, w, b, tm, em, task)[1] <= task.risk_cap
            li, lw, _ = choose("uncertain", b, m.pi, tm, em, task, u_trigger=0.1, legacy=True)
            found |= lw == 3 and cost(li, lw, b, tm, em, task)[1] > task.risk_cap
    assert found  # K's behaviour exceeded the cap on some of these beliefs


# ---- A6: the change model (TypeStateModel) -------------------------------------------------------------------------

def _two_type(lam=0.1, **kw):
    import numpy as np

    from apm_revise.belief import TypeModel
    from apm_transfer.nerva_memory import TypeStateModel
    calm = np.array([0.7, 0.2, 0.05, 0.05])
    push = np.array([0.05, 0.05, 0.2, 0.7])
    m = TypeModel("t", np.array([0.8, 0.2]), np.vstack([calm, push]), np.array([0.1, 0.9]))
    return TypeStateModel(m, lam=lam, transition_each_meeting=True, **kw)


def test_a6_lambda_applies_exactly_once_per_encounter():
    import math

    import numpy as np
    for log_space in (False, True):
        tm = _two_type(lam=0.2, log_space=log_space)
        tm.observe_many("a", "x", [3], 0.0)
        b1 = tm.b["a"].copy()
        pi = tm.model.pi
        # next encounter with "a" right away: one transition (predict) before the update
        assert np.allclose(tm.belief("a", "x"), pi + (b1 - pi) * math.exp(-0.2 * 1))
        tm.observe_many("b", "x", [0], 0.0)          # an encounter with another identity
        tm.observe_many("a", "x", [], 0.0)           # an encounter with "a", nothing observable
        assert np.allclose(tm.b["a"], b1)            # nothing learned
        assert np.allclose(tm.belief("a", "x"), pi + (b1 - pi) * math.exp(-0.2 * 3))  # 2 elapsed + the coming one


def test_a6_consecutive_and_alternating_identities_agree_on_the_per_encounter_clock():
    import numpy as np
    seq = [("a", 3), ("b", 0), ("a", 3), ("b", 0), ("a", 0), ("a", 0)]
    tm = _two_type()
    for ident, c in seq:
        before = tm.belief(ident, "x")
        tm.observe_many(ident, "x", [c], 0.0)
        post = before * tm.model.E[:, c]
        assert np.allclose(tm.b[ident], post / post.sum())   # update = predicted belief x likelihood


def test_a6_log_space_equals_sequential_product_and_survives_strong_evidence():
    import numpy as np
    seq_tm, log_tm = _two_type(), _two_type(log_space=True)
    for tm in (seq_tm, log_tm):
        tm.observe_many("a", "x", [3, 3, 2, 3], 0.0)
    assert np.allclose(seq_tm.b["a"], log_tm.b["a"], rtol=1e-12, atol=1e-15)
    tm = _two_type(log_space=True)
    tm.observe_many("a", "x", [3] * 400, 0.0)      # ratio (0.05/0.7)^400 ~ 1e-459: underflows double precision
    assert np.all(np.isfinite(tm.b["a"])) and tm.b["a"][1] == 1.0
    for _ in range(60):                             # it still revises: the transition restores mass, evidence moves it
        tm.observe_many("a", "x", [0, 0], 0.0)
    assert tm.belief("a", "x")[0] > 0.5


def test_a6_near_zero_posterior_recovers_in_both_directions():
    import numpy as np
    for start, then, target in ((3, 0, 0), (0, 3, 1)):  # pushy->calm and calm->pushy
        tm = _two_type()
        for _ in range(20):
            tm.observe_many("a", "x", [start] * 3, 0.0)
        assert tm.b["a"][target] < 1e-3   # the per-meeting transition keeps it from reaching exactly 0
        n = 0
        while tm.belief("a", "x")[target] < 0.5:
            tm.observe_many("a", "x", [then] * 3, 0.0)
            n += 1
            assert n < 10, (start, then)
        assert np.isfinite(tm.belief("a", "x")).all()


def test_a6_zero_likelihood_everywhere_keeps_the_prediction_in_log_space():
    import numpy as np
    tm = _two_type(log_space=True)
    tm.model.E[:, 1] = 0.0
    tm.observe_many("a", "x", [1], 0.0)
    assert np.allclose(tm.b["a"], tm.model.pi)


def test_a6_save_restore_is_exact():
    import json

    import numpy as np
    seq = [("a", [3]), ("b", [0, 1]), ("a", []), ("c", [2, 3, 3]), ("a", [0])]
    ref, tm = _two_type(), _two_type()
    for ident, cs in seq[:3]:
        ref.observe_many(ident, "x", cs, 0.4)
        tm.observe_many(ident, "x", cs, 0.4)
    restored = _two_type()
    restored.load_state(json.loads(json.dumps(tm.state())))
    for ident, cs in seq[3:]:
        ref.observe_many(ident, "x", cs, 0.4)
        restored.observe_many(ident, "x", cs, 0.4)
    for ident in "abc":
        assert np.array_equal(ref.belief(ident, "x"), restored.belief(ident, "x"))
    assert ref.n == restored.n and ref.g == restored.g


def test_a6_legacy_clock_reproduces_the_d1_defect():
    """K3-X's memory: an identity met in consecutive encounters never relaxes, so certainty cannot be revised."""
    import numpy as np

    from apm_transfer.nerva_memory import TypeStateModel
    tm = _two_type()
    legacy = TypeStateModel(tm.model, lam=0.1, transition_each_meeting=False)
    for _ in range(200):
        legacy.observe_many("a", "x", [3] * 3, 0.0)
    assert legacy.b["a"][0] < 1e-200
    for _ in range(9):
        legacy.observe_many("a", "x", [0] * 3, 0.0)
    assert legacy.belief("a", "x")[0] < 0.5
    assert np.isfinite(legacy.b["a"]).all()


def test_v1_observe_and_hold_never_complete_a_movement_task():
    import numpy as np

    from apm_transfer.expression_map import TimeModel
    from apm_transfer.infoseek import HOLD, OBSERVE, Emissions, Task, action_cost
    m = _all_risky_model()
    tm = TimeModel([20.0, 10.0, -17.0, -1.5, 6.5])
    em = Emissions.from_model(m.E, camera_block=True)
    b = np.array([0.5, 0.5])
    for name, kw in (("waypoint", {"d_task": None}), ("separation", {"d_task": None, "min_sep": 0.5})):
        new = Task(name, retry_s=12.6, **kw)
        old = Task(name, retry_s=12.6, hold_completes=True, **kw)
        for a in (OBSERVE, HOLD):
            assert action_cost(a, 1, b, tm, em, new)[0] - action_cost(a, 1, b, tm, em, old)[0] == 12.6
