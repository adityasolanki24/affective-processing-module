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
