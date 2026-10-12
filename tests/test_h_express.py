"""Programme H expressive interface: bounded intents, adapter constraints, independent monitor, fallback."""
from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from apm_express.adapter import COMMAND_NAMES, D_FAR, D_NEAR, HEAD_EPS, BipedChannels, BipedExpressionAdapter
from apm_express.episode import EpisodeSpec, run_episode
from apm_express.intent import CHANNELS, NEUTRAL, RANGES, ExpressionIntent, blend
from apm_express.kinematic import KinematicBiped
from apm_express.safety import SafetyMonitor

H = Path(__file__).resolve().parents[1] / "experiments" / "H_expressive_intelligence"


def _random_intent(rng) -> ExpressionIntent:
    values = rng.uniform(-3, 3, len(CHANNELS))
    values[rng.random(len(CHANNELS)) < 0.1] = np.nan
    values[rng.random(len(CHANNELS)) < 0.05] = np.inf
    return ExpressionIntent(**dict(zip(CHANNELS, values.tolist(), strict=True)))


def test_intent_sanitized_is_bounded_and_finite():
    rng = np.random.default_rng(0)
    for _ in range(200):
        it = _random_intent(rng).sanitized()
        for c in CHANNELS:
            v = getattr(it, c)
            lo, hi = RANGES[c]
            assert np.isfinite(v) and lo <= v <= hi
    assert ExpressionIntent(tempo=float("nan")).sanitized().tempo == NEUTRAL["tempo"]
    with pytest.raises(ValueError):
        ExpressionIntent.from_array([0.0, 1.0])
    mid = blend(ExpressionIntent(lean=-1), ExpressionIntent(lean=1), 0.5)
    assert abs(mid.lean) < 1e-12


@pytest.mark.parametrize("head", [False, True])
def test_adapter_outputs_stay_inside_validated_channels(head):
    rng = np.random.default_rng(1)
    ch = BipedChannels(min_dwell_s=0.5, pitch=True, pitch_rate=1.0, head=head, head_pitch_range=(-0.2, 0.3),
                       head_yaw_range=(-0.4, 0.4), head_rate=0.8)
    ad = BipedExpressionAdapter(ch, seed=3)
    ctrl = KinematicBiped()
    ad.reset(0.0, ctrl.pose(), (1.0, 0.5), ExpressionIntent())
    last_cmd, since, last_pitch, last_head = None, 0.0, 0.0, np.zeros(4)
    intent = _random_intent(rng)
    for k in range(3000):
        if k % 100 == 0:  # the intent changes abruptly and arbitrarily
            intent = _random_intent(rng)
        t = ctrl.t
        target = None if (k // 700) % 3 == 2 else (1.0 + 0.2 * (k // 700), 0.5)
        req = ad.request(t, ctrl.pose(), target, intent)
        assert req.command in COMMAND_NAMES
        if last_cmd is not None and req.command != last_cmd:
            assert t - since >= ch.min_dwell_s - 1e-9
            since = t
        last_cmd = req.command if last_cmd is None or req.command != last_cmd else last_cmd
        assert -1.0 <= req.e_pitch <= 1.0 and abs(req.e_pitch - last_pitch) <= ch.pitch_rate * 0.02 + 1e-12
        h = np.array(req.head)
        assert np.all(np.abs(h - last_head) <= ch.head_rate * 0.02 + 1e-12)
        if head:
            assert ch.head_pitch_range[0] - 1e-12 <= h[1] <= ch.head_pitch_range[1] + 1e-12
            assert ch.head_yaw_range[0] - 1e-12 <= h[2] <= ch.head_yaw_range[1] + 1e-12
            assert h[0] == 0.0 and h[3] == 0.0
            if req.command != "rest":  # head offsets only while resting (head_while_moving=False)
                assert np.max(np.abs(h)) <= HEAD_EPS + 1e-12
        else:
            assert np.all(h == 0.0)
        last_pitch, last_head = req.e_pitch, h
        ctrl.apply(req)
        ctrl.advance()


def test_pitch_channel_disabled_means_zero_pitch():
    ad = BipedExpressionAdapter(BipedChannels(pitch=False), seed=0)
    req = ad.request(0.0, (0, 0, 0), (1.0, 0.0), ExpressionIntent(lean=1.0))
    assert req.e_pitch == 0.0


@pytest.mark.parametrize("proximity", [0.0, 0.5, 1.0])
def test_neutral_style_reaches_its_stopping_distance(proximity):
    ctrl = KinematicBiped()
    ad = BipedExpressionAdapter(BipedChannels(min_dwell_s=0.5), seed=0)
    spec = EpisodeSpec(40.0, (1.2, 0.3), [(0.0, ExpressionIntent(proximity=proximity))])
    r = run_episode(ctrl, ad, SafetyMonitor(), spec)
    a = r["arrays"]
    d = math.hypot(1.2 - a["x"][-1], 0.3 - a["y"][-1])
    d_stop = D_FAR - (D_FAR - D_NEAR) * proximity
    assert d <= d_stop + 0.05 and d >= d_stop - 0.15
    assert r["trips"] == 0 and r["fallback_steps"] == 0


def test_monitor_trip_and_resume_rule():
    m = SafetyMonitor()
    assert not m.update(0.0, 10.0, 1.0)
    assert m.update(0.02, 25.0, 0.0) and m.trips == 1
    assert m.update(0.04, 13.0, 0.0)  # not yet calm
    t = 0.06
    while t < 0.06 + 0.5 - 1e-9:
        assert m.update(t, 11.0, 0.0)
        t = round(t + 0.02, 10)
    assert not m.update(t, 11.0, 0.0)
    assert m.update(t + 0.02, 0.0, 5.0) and m.trips == 2  # angular-speed rule (NERVA risk 0.75)
    assert SafetyMonitor().update(0.0, float("nan"), 0.0)


def test_monitor_never_sees_expressive_inputs():
    params = set(inspect.signature(SafetyMonitor.update).parameters)
    assert params == {"self", "t", "tilt_deg", "angular_speed", "joint_pos", "joint_lo", "joint_hi",
                      "actuator_force", "force_limit"}


def test_fallback_sends_neutral_rest_and_zero_offsets():
    ctrl = KinematicBiped()
    ad = BipedExpressionAdapter(BipedChannels(min_dwell_s=0.5, pitch=True), seed=0)
    intent = ExpressionIntent(lean=1.0, scanning=1.0, tempo=1.0)

    class Tilting(KinematicBiped):
        def physical(self):
            out = super().physical()
            out["tilt_deg"] = 30.0 if 2.0 <= self.t < 2.5 else 0.0
            return out

    ctrl = Tilting()
    r = run_episode(ctrl, ad, SafetyMonitor(), EpisodeSpec(6.0, (2.0, 0.0), [(0.0, intent)]))
    a = r["arrays"]
    fb = a["fallback"].astype(bool)
    assert fb.any() and r["trips"] == 1
    assert np.all(a["command"][fb] == COMMAND_NAMES.index("rest"))
    assert np.all(a["e_pitch"][fb] == 0.0) and np.all(a["head_request"][fb] == 0.0)
    first_after = np.argmax(~fb & (np.arange(len(fb)) > np.argmax(fb)))
    assert abs(a["e_pitch"][first_after]) <= 0.02 + 1e-12  # restarts from neutral, rate-limited


def test_h_seed_partitions_are_disjoint_from_earlier_programmes():
    spec = importlib.util.spec_from_file_location("h_seeds", H / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("h_seeds", mod)
    spec.loader.exec_module(mod)
    states = {mod.fingerprint(mod.world_seed(e, p, i))
              for e in mod.EXPERIMENTS for p in mod.PARTITIONS for i in range(3)}
    assert len(states) == len(mod.EXPERIMENTS) * len(mod.PARTITIONS) * 3
    for entropy in mod.EARLIER_PROGRAMME_ENTROPIES:
        for e in range(9):
            for p in range(1, 6):
                old = np.random.SeedSequence(entropy, spawn_key=(e, p, 0))
                assert mod.fingerprint(old) not in states
    assert 0 <= mod.sim_seed("H0", "X", 0) < 2**31
