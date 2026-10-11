"""G6 variations: physics patches are restored, input transforms behave as specified, every condition builds."""
from __future__ import annotations

import numpy as np

import apm_testbeds.wheeled_ctrl as W
from apm_research.harness import run_lifetime
from apm_research.policies import B0
from apm_select.variations import CONDITIONS, InputTransform, lifetime, patched


def test_physics_patch_is_temporary():
    before = (W.ACCEL, dict(W.MODES))
    with patched("slow_actuation"):
        assert W.ACCEL == 0.5
    with patched("speed_limit_0.6"):
        assert W.MODES[W.GO][0] == 0.6
    assert (W.ACCEL, dict(W.MODES)) == before


def test_speed_limit_changes_motion():
    ss = np.random.SeedSequence(5)
    fast = run_lifetime(lifetime("nominal", ss, 1), B0())
    with patched("speed_limit_0.6"):
        slow = run_lifetime(lifetime("nominal", ss, 1), B0())
    assert slow["distance"] < fast["distance"]


class _Rec:
    name = "rec"
    def __init__(self):
        self.seen = []
    def begin_lifetime(self): ...
    def begin_episode(self, t0): ...
    def observe(self, obs, t):
        self.seen.append((obs.t, t, obs))
    def decide(self, t):
        return 0
    def end_episode(self, t): ...


def test_delay_and_noise_transforms():
    ss = np.random.SeedSequence(7)
    plain, delayed, noisy = _Rec(), _Rec(), _Rec()
    run_lifetime(lifetime("nominal", ss, 1), plain)
    run_lifetime(lifetime("nominal", ss, 1), InputTransform(delayed, "delay", 5))
    run_lifetime(lifetime("nominal", ss, 1), InputTransform(noisy, "noise", 0.1, seed=1))
    assert len(delayed.seen) == len(plain.seen) - 5
    for (ot, t, _), (pt, _, _) in zip(delayed.seen, plain.seen, strict=False):
        assert ot == pt and t > ot  # stale state delivered 5 steps later
    diffs = [abs(a.distance - b.distance) for (_, _, o1), (_, _, o2) in zip(plain.seen, noisy.seen, strict=True)
             for a, b in zip(o1.entities, o2.entities, strict=True)]
    assert np.mean(diffs) > 0.03


def test_every_condition_builds_a_world():
    for c in CONDITIONS:
        lt = lifetime(c, np.random.SeedSequence(3), 1)
        with patched(c):
            assert lt.world(0) is not None
