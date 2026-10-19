"""Expression v2: label-free experience memory and the designed expressive program."""
import numpy as np
import pytest

from apm_isaac.posture import RateLimited, nerva_offsets, t1_offsets
from apm_transfer.experience_state import DisturbanceMemory, ExperienceState, calibrate_threshold
from apm_transfer.expressive_program import plan, upper_targets


def test_threshold_is_label_free_and_above_standing_sway():
    sway = list(np.random.default_rng(0).uniform(0.0, 0.02, 200))
    th = calibrate_threshold(sway)
    assert th > max(sway) * 0.98
    with pytest.raises(ValueError):
        calibrate_threshold([0.01] * 5)


def test_disturbance_memory_learns_and_revises():
    m = DisturbanceMemory(threshold_m_s=0.05)
    for _ in range(8):
        m.observe("B1", [0.01, 0.12])  # pushed during a window
    high = m.probability("B1")
    for _ in range(8):
        m.observe("B1", [0.01, 0.02])  # calm meetings: the outdated tendency fades
    assert high > 0.75 and m.probability("B1") < 0.35
    m.observe("B1", None)  # no valid window is not evidence
    m.observe("C1", [])
    assert "C1" not in m.counts


def test_histories_give_different_states_from_sensing_only():
    pushy, calm = ExperienceState(), ExperienceState()
    for _ in range(8):
        pushy.observe("B1", [0.02, 0.15], [[0.25]])
        calm.observe("B1", [0.02, 0.01], [[0.01]])
    a, b = pushy.readout("B1"), calm.readout("B1")
    assert a["wariness"] - b["wariness"] > 0.4
    assert a["familiarity"] == pytest.approx(b["familiarity"])


def test_experience_persists_through_state_round_trip():
    e = ExperienceState()
    e.observe("B1", [0.2], [[0.3]])
    f = ExperienceState()
    f.load(e.state())
    assert f.readout("B1") == e.readout("B1")


def test_plan_is_monotone_in_wariness_and_matched_removes_speed_and_distance():
    lo = plan({"wariness": 0.1, "uncertainty": 0.2, "familiarity": 0.9})
    hi = plan({"wariness": 0.9, "uncertainty": 0.2, "familiarity": 0.9})
    assert hi.observe_s > lo.observe_s and hi.guard > lo.guard and hi.retreat_m > lo.retreat_m
    assert hi.directness < lo.directness and lo.openness > hi.openness
    lm = plan({"wariness": 0.1, "uncertainty": 0.2, "familiarity": 0.9}, matched=True)
    hm = plan({"wariness": 0.9, "uncertainty": 0.2, "familiarity": 0.9}, matched=True)
    assert (lm.tempo, lm.proximity, lm.hesitation) == (hm.tempo, hm.proximity, hm.hesitation)
    assert hm.guard != lm.guard and hm.observe_s != lm.observe_s


def test_neutral_program_has_no_expression():
    p = plan(None)
    g = upper_targets(p, "interact", 1.0, 0.3)
    assert g["guard"] == g["openness"] == g["tilt"] == g["side_on"] == 0.0
    # while sensing every program looks at the person (functional, for the camera), not expressive
    assert g["gaze_yaw"] == 0.3 and upper_targets(p, "approach", 1.0, 0.3)["gaze_yaw"] == 0.0


def test_body_offsets_stay_in_bounds_and_rate_limited():
    for g in ({"gaze_yaw": 5, "gaze_pitch": 5, "guard": 1, "openness": 1, "tilt": 1, "side_on": 1},
              {"gaze_yaw": -5, "gaze_pitch": -5, "guard": 0, "openness": 0, "tilt": 0, "side_on": 0}):
        n = nerva_offsets(g)
        assert np.all(np.abs(n[1:3]) <= 0.4 + 1e-9) and abs(n[0]) <= 0.25 and abs(n[3]) <= 0.25
        t = t1_offsets(g)
        assert abs(t[0]) <= 0.6 and abs(t[1]) <= 0.3 and abs(t[10]) <= 0.35
    r = RateLimited(4, 0.75)
    v = r.step(np.ones(4), 0.02)
    assert np.allclose(v, 0.015)
    assert np.allclose(r.step(np.ones(4), 0.02, allowed=False), 0.0)


def test_memory_routing_controls():
    from apm_transfer.memory_routing import POOLED, decide_identity, observe_identity
    ents = (("A1", "A", "P1"), ("B1", "B", "P2"), ("C1", "C", "P3"))
    rng = np.random.default_rng(0)
    assert decide_identity("learn", "B1") == observe_identity("learn", "B1", ents, rng) == "B1"
    assert decide_identity("pooled", "A1") == observe_identity("pooled", "B1", ents, rng) == POOLED
    # swapped reads the other person's memory but writes the true person's: not a mere renaming
    assert decide_identity("swapped", "B1") == "A1" and observe_identity("swapped", "B1", ents, rng) == "B1"
    assert {observe_identity("shuffled", "B1", ents, rng) for _ in range(50)} == {"A1", "B1", "C1"}
    with pytest.raises(ValueError):
        observe_identity("shuffled", "B1", ents, None)


def test_looming_detects_an_approach_but_not_quiet_frames():
    from apm_isaac.world import _looming_peak
    lunge = [(0.0, 0.30), (0.1, 0.29), (0.2, 0.12), (0.3, 0.05), (0.9, 0.31)]
    quiet = [(0.0, 0.30), (0.1, 0.31), (0.2, 0.30), (0.6, 0.29)]
    assert _looming_peak(lunge) == pytest.approx(0.25) and _looming_peak(quiet) <= 0.021
    assert _looming_peak([(0.0, 0.5), (0.8, 0.1)]) == 0.0  # slower than 0.5 s is not a lunge


def test_pooled_looming_sees_an_approach_anywhere_but_not_a_static_floor():
    from apm_isaac.world import _looming_peak, _pool_depth
    rng = np.random.default_rng(0)
    floor = np.tile(np.linspace(0.2, 1.5, 48)[:, None], (1, 64))  # static scene, near floor at the bottom
    quiet = [(0.1 * i, _pool_depth(floor + rng.normal(0, 0.003, floor.shape))) for i in range(6)]
    lunge = [(0.1 * i, _pool_depth(np.where(np.arange(64)[None, :] > 40, 1.0 - 0.15 * i, floor)
                                   + rng.normal(0, 0.003, floor.shape))) for i in range(6)]
    assert _looming_peak(quiet) < 0.02 and _looming_peak(lunge) > 0.3
