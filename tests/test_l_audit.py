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
