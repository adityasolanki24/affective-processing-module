"""Programme L (audit A1, A2): event times that require *confirmed* evidence.

`confirmed_onset(ok, need)`:
- The event occurs at the first index j such that ok[j], ok[j+1], …, ok[j+need-1] are all True, **and all of those
  observations exist**.
- A favourable observation on the last available meeting is unconfirmed, so the lifetime is right-censored at the
  number of observations.
- This is the definition programme K's protocol states ("for that meeting and the next"). K's `k_analyze.py`
  counted a lone favourable final meeting as an event; `legacy=True` reproduces that, for checking K's numbers only.

Returns (time, event): time is 1-based in observations; event False means censored.
"""
from __future__ import annotations

from collections.abc import Sequence


def confirmed_onset(ok: Sequence[bool], need: int = 2, legacy: bool = False) -> tuple[float, bool]:
    n = len(ok)
    for j in range(n):
        if legacy:
            if ok[j] and (j + 1 == n or ok[j + 1]):
                return float(j + 1), True
            continue
        if j + need <= n and all(ok[j:j + need]):
            return float(j + 1), True
    return float(max(1, n)), False


def within(value: float, target: float, tol: float) -> bool:
    return abs(value - target) <= tol


def reached(value: float, limit: float, tol: float = 0.0) -> bool:
    """Task-derived criterion: the stopping distance is at most the task's interaction distance plus a tolerance."""
    return value <= limit + tol
