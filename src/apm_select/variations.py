"""G6 held-out procedural variations (never used for training or tuning anywhere in programme G).

Three mechanisms, none of which edits the frozen testbed's code:
  world   — WorldConfig / population settings (obstacle and crowd density, pedestrian speed, hazard mix, friction,
            identity errors, deadlines);
  input   — transforms of what the policy receives (estimator miscalibration, sensor noise, observation delay);
  physics — temporary values of the testbed's module constants for the duration of one job (acceleration limit,
            GO speed cap), restored afterwards (`patched`).
The simulator, navigation and safety supervisor code are unchanged; the policy is never told the condition.
Not varied: arena size (layout coordinates are hard-coded in the frozen testbed).
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
from collections import deque

import numpy as np

import apm_testbeds.wheeled_ctrl as W
from apm_research.harness import Lifetime
from apm_testbeds.wheeled_ctrl import GO, WorldConfig, ttc_probability

CONDITIONS: dict[str, dict] = {
    "nominal": {},
    "obstacles_dense": {"world": {"obstacles": (7, 10)}},
    "crowd_dense": {"world": {"people": (4, 6)}, "n_people": 8},
    "fast_pedestrians": {"world": {"ped_speed": (0.8, 1.6)}},
    "careless_heavy": {"p_careless": 0.8},
    "slippery": {"world": {"patches": (3, 6)}},
    "identity_uncertain": {"world": {"identity_error": 0.3, "identity_ambiguous": 0.2}},
    "deadline_tight": {"world": {"deadline_base_s": 2.0, "deadline_speed": 0.45}},
    "estimator_x0.5": {"input": ("scale", 0.5)},
    "estimator_x3": {"input": ("scale", 3.0)},
    "sensor_noise": {"input": ("noise", 0.10)},
    "obs_delay_0.25s": {"input": ("delay", 5)},
    "slow_actuation": {"physics": {"ACCEL": 0.5}},
    "speed_limit_0.6": {"physics": {"GO_CAP": 0.6}},
}


def lifetime(condition: str, ss, n_episodes: int) -> Lifetime:
    c = CONDITIONS[condition]
    cfg = dataclasses.replace(WorldConfig(), **c.get("world", {}))
    return Lifetime(ss, n_episodes=n_episodes, config=cfg, n_people=c.get("n_people", 6),
                    p_careless=c.get("p_careless", 0.4))


@contextlib.contextmanager
def patched(condition: str):
    """Temporarily set physics constants for one job (worker processes run one job at a time)."""
    phys = CONDITIONS[condition].get("physics", {})
    saved = (W.ACCEL, dict(W.MODES))
    try:
        if "ACCEL" in phys:
            W.ACCEL = phys["ACCEL"]
        if "GO_CAP" in phys:
            _, rng, gain = W.MODES[GO]
            W.MODES[GO] = (phys["GO_CAP"], rng, gain)
        yield
    finally:
        W.ACCEL = saved[0]
        W.MODES.clear()
        W.MODES.update(saved[1])


class InputTransform:
    """Wraps a policy; transforms observations before the policy sees them (scale / noise / delay)."""

    def __init__(self, policy, kind: str, value, seed: int = 0):
        self.policy, self.kind, self.value = policy, kind, value
        self.rng = np.random.default_rng(seed)
        self.buffer: deque = deque()

    def __getattr__(self, item):
        return getattr(self.policy, item)

    def begin_lifetime(self) -> None:
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.buffer.clear()
        self.policy.begin_episode(t0)

    def _noisy(self, obs):
        s = self.value
        ents = []
        for e in obs.entities:
            d = max(0.0, e.distance + self.rng.normal(0, s))
            b = e.bearing + self.rng.normal(0, s)
            c = e.closing + self.rng.normal(0, s)
            ents.append(dataclasses.replace(e, distance=d, bearing=math.atan2(math.sin(b), math.cos(b)), closing=c,
                                            ttc_p=ttc_probability(d, c)))
        return dataclasses.replace(obs, entities=tuple(ents))

    def observe(self, obs, t: float) -> None:
        if self.kind == "scale":
            obs = dataclasses.replace(obs, entities=tuple(dataclasses.replace(e, ttc_p=min(1.0, self.value * e.ttc_p))
                                                          for e in obs.entities))
            self.policy.observe(obs, t)
        elif self.kind == "noise":
            self.policy.observe(self._noisy(obs), t)
        elif self.kind == "delay":
            self.buffer.append((obs, t))
            if len(self.buffer) > self.value:
                old, _ = self.buffer.popleft()
                # stale state delivered now; events are delivered late with it (no event is lost or duplicated)
                self.policy.observe(old, t)
        else:
            raise ValueError(self.kind)

    def decide(self, t: float) -> int:
        return self.policy.decide(t)

    def end_episode(self, t: float) -> None:
        self.policy.end_episode(t)


def wrap(policy, condition: str, seed: int = 0):
    inp = CONDITIONS[condition].get("input")
    return InputTransform(policy, inp[0], inp[1], seed) if inp else policy
