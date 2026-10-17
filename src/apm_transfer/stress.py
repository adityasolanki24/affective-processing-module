"""J2 stress conditions for the wheeled testbed (MASTER_PREREGISTRATION §3.1). The frozen testbed's code is not edited.

Mechanisms:
- **world settings:** `WorldConfig` fields, population size and careless share;
- **pedestrian behaviour:** a `WheeledCtrlWorld` subclass. Before each movement step it swaps a person's *acting*
  disposition; the identity object is restored immediately afterwards, so collision logs and truth keep the identity's
  own disposition. Behaviours:
  - `overlap`: on each approach (gap falls below 1.5 m) the person acts with the other disposition with probability q,
    until the gap exceeds 2 m;
  - `erratic`: designated identities (40% of the population) switch acting disposition every 1–3 s at random. This
    pattern is never seen in training;
  - `drift`: in S1/S2 the switched identities act with the post-switch disposition with probability
    clip((e − 10)/10, 0, 1) per approach, instead of switching abruptly at episode 15. Truth still switches at 15;
- **sensing:** `apm_select.variations.InputTransform`, so the noise reaches everything the robot (and its memory)
  observes;
- **robot speed:** module constants patched for one job (`speed_limit_0.6`, `robot_fast`).

`make_stress(condition, scenario, seed, episodes)` → (lifetime, roles, policy wrapper, context manager or None).
"""
from __future__ import annotations

import contextlib
import dataclasses
import math

import numpy as np

import apm_testbeds.wheeled_ctrl as W
from apm_research.harness import Lifetime
from apm_revise.scenarios import make_lifetime
from apm_select.variations import InputTransform
from apm_testbeds.wheeled_ctrl import GO, Identity, WheeledCtrlWorld, child

OTHER = {"careless": "careful", "careful": "careless"}

CONDITIONS: dict[str, dict] = {
    "nominal": {},
    "overlap_25": {"behaviour": {"overlap": 0.25}},
    "overlap_40": {"behaviour": {"overlap": 0.40}},
    "erratic": {"behaviour": {"erratic": 0.40}},
    "drift": {"behaviour": {"drift": True}},
    "identity_uncertain": {"world": {"identity_error": 0.3, "identity_ambiguous": 0.2}},
    "sensor_noise_0.10": {"input": ("noise", 0.10)},
    "sensor_noise_0.20": {"input": ("noise", 0.20)},
    "crowd_dense": {"world": {"people": (4, 6)}, "n_people": 8},
    "obstacles_dense": {"world": {"obstacles": (7, 10)}},
    "fast_pedestrians": {"world": {"ped_speed": (0.8, 1.6)}},
    "speed_limit_0.6": {"physics": {"go_cap": 0.6}},
    "robot_fast": {"physics": {"go_cap": 1.0, "v_max": 1.0}},
}


class StressWorld(WheeledCtrlWorld):
    def __init__(self, seed, population, config, behaviour: dict, episode: int, erratic: frozenset,
                 drift_ids: dict[str, tuple[str, str]]):
        super().__init__(seed, population, config)
        self.behaviour, self.episode, self.erratic, self.drift_ids = behaviour, episode, erratic, drift_ids
        self.brng = np.random.default_rng(child(seed, 7))
        self.acting: dict[str, str] = {}      # body id → acting disposition during the current approach
        self.next_flip: dict[str, float] = {}

    def _acting(self, o) -> str:
        base = o.identity.disposition
        name = o.identity.name
        gap = math.hypot(self.x - o.x, self.y - o.y) - o.r - W.ROBOT_R
        if name in self.erratic:
            if self.t >= self.next_flip.get(o.oid, -1.0):
                self.acting[o.oid] = "careless" if self.brng.random() < 0.5 else "careful"
                self.next_flip[o.oid] = self.t + float(self.brng.uniform(1.0, 3.0))
            return self.acting[o.oid]
        q = self.behaviour.get("overlap", 0.0)
        drift = self.drift_ids.get(name)
        if q or drift:
            if gap < 1.5 and o.oid not in self.acting:
                if drift:
                    pre, post = drift
                    p_post = min(1.0, max(0.0, (self.episode - 10) / 10.0))
                    self.acting[o.oid] = post if self.brng.random() < p_post else pre
                else:
                    self.acting[o.oid] = OTHER[base] if self.brng.random() < q else base
            elif gap > 2.0:
                self.acting.pop(o.oid, None)
            return self.acting.get(o.oid, base)
        return base

    def _move_people(self) -> None:
        saved = []
        for o in self.bodies:
            if o.kind == "person" and o.identity is not None:
                act = self._acting(o)
                if act != o.identity.disposition:
                    saved.append((o, o.identity))
                    o.identity = Identity(o.identity.name, act)
        try:
            super()._move_people()
        finally:
            for o, ident in saved:
                o.identity = ident


@dataclasses.dataclass
class StressLifetime(Lifetime):
    behaviour: dict = dataclasses.field(default_factory=dict)
    roles: dict = dataclasses.field(default_factory=dict)

    def erratic_ids(self) -> frozenset:
        share = self.behaviour.get("erratic", 0.0)
        if not share:
            return frozenset()
        names = [i.name for i in self.population()]
        rng = np.random.default_rng(child(self.seed, 99))
        k = max(1, int(round(share * len(names))))
        return frozenset(str(n) for n in rng.choice(names, size=k, replace=False))

    def world(self, episode: int) -> WheeledCtrlWorld:
        base = self.population()
        pop = self.schedule(episode, base) if self.schedule else base
        drift_ids: dict[str, tuple[str, str]] = {}
        if self.behaviour.get("drift") and self.roles.get("switched"):
            to = self.roles["to"]
            for name in self.roles["switched"]:
                drift_ids[name] = (OTHER[to], to)
        return StressWorld(child(self.seed, 1 + episode), pop, self.config, self.behaviour, episode,
                           self.erratic_ids(), drift_ids)


@contextlib.contextmanager
def _physics(phys: dict):
    saved = (dict(W.MODES), W.V_MAX)
    try:
        if "go_cap" in phys:
            _, rng, gain = W.MODES[GO]
            W.MODES[GO] = (phys["go_cap"], rng, gain)
        if "v_max" in phys:
            W.V_MAX = phys["v_max"]
        yield
    finally:
        W.MODES.clear()
        W.MODES.update(saved[0])
        W.V_MAX = saved[1]


def make_stress(condition: str, scenario: str, ss, episodes: int = 30):
    c = CONDITIONS[condition]
    lt, roles = make_lifetime(scenario, ss, episodes)
    cfg = dataclasses.replace(lt.config, **c.get("world", {}))
    slt = StressLifetime(lt.seed, lt.n_episodes, cfg, c.get("n_people", lt.n_people), lt.p_careless, lt.schedule,
                         behaviour=c.get("behaviour", {}), roles=roles)
    roles = {**roles, "condition": condition}
    if slt.erratic_ids():
        roles["erratic"] = sorted(slt.erratic_ids())
    inp = c.get("input")
    seed = int(np.random.default_rng(child(ss, 98)).integers(2**31))
    wrap = (lambda p: InputTransform(p, inp[0], inp[1], seed)) if inp else (lambda p: p)
    ctx = (lambda: _physics(c["physics"])) if "physics" in c else None
    return slt, roles, wrap, ctx
