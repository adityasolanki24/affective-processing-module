"""G2 scenario families (30 × 60 s episodes per lifetime, population of 6 identities unless stated).

Same constructions as programme F's E3 (`F/scripts/e3_run.py`), re-implemented here so F's code stays frozen, plus
new families. Ground truth (`schedule`, `roles`) is for evaluation only.

  S1  risky → safe:  ped-0, ped-1 careless until episode 15, then careful
  S2  safe → risky:  ped-0, ped-1 careful until episode 15, then careless
  S3  stable mixed:  ped-0 careless, ped-1 careful, others drawn (40% careless)
  S4  identity errors: S3 with 20% wrong reported identities and 10% ambiguous (confidence 0.4)
  S5  confounded co-occurrence: careful ped-0 frequents low-friction patches (slips co-occur without contact)
  S6  previously avoided: careless ped-0 always present
  S7  newcomer: careless-heavy history, a careful newcomer ped-5 from episode 15
  S8  frequent vs rare: careless ped-0 always present; careless ped-5 present only in every fifth episode
  S9  sensor gaps: S3, and the policy receives person sightings without identity in 2 s gaps every 6 s
"""
from __future__ import annotations

import dataclasses

from apm_research.harness import Lifetime
from apm_testbeds.wheeled_ctrl import Identity, WorldConfig

EPISODES, SWITCH = 30, 15
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")


def _with(base: list[Identity], fixed: dict[str, str]) -> list[Identity]:
    return [Identity(i.name, fixed.get(i.name, i.disposition)) for i in base]


def make_lifetime(scenario: str, ss, episodes: int = EPISODES) -> tuple[Lifetime, dict]:
    cfg, roles = WorldConfig(), {}
    if scenario == "S1":
        def sched(e, base):
            d = "careless" if e < SWITCH else "careful"
            return _with(base, {"ped-0": d, "ped-1": d})
        roles = {"switched": ["ped-0", "ped-1"], "to": "careful"}
    elif scenario == "S2":
        def sched(e, base):
            d = "careful" if e < SWITCH else "careless"
            return _with(base, {"ped-0": d, "ped-1": d})
        roles = {"switched": ["ped-0", "ped-1"], "to": "careless"}
    elif scenario in ("S3", "S4", "S9"):
        def sched(e, base):
            return _with(base, {"ped-0": "careless", "ped-1": "careful"})
        if scenario == "S4":
            cfg = WorldConfig(identity_error=0.2, identity_ambiguous=0.1)
        if scenario == "S9":
            roles = {"sensor_gaps": True}
    elif scenario == "S5":
        cfg = WorldConfig(patches=(1, 3), patch_affinity=("ped-0",), force_present=("ped-0",))
        def sched(e, base):
            return _with(base, {"ped-0": "careful", "ped-1": "careful"})
        roles = {"affine": "ped-0"}
    elif scenario == "S6":
        cfg = WorldConfig(force_present=("ped-0",))
        def sched(e, base):
            return _with(base, {"ped-0": "careless"})
        roles = {"avoided": "ped-0"}
    elif scenario == "S7":
        def sched(e, base):
            pop = _with(base, {"ped-0": "careless", "ped-1": "careless", "ped-2": "careless", "ped-3": "careless",
                               "ped-4": "careful", "ped-5": "careful"})
            return pop if e >= SWITCH else [i for i in pop if i.name != "ped-5"]
        roles = {"newcomer": "ped-5"}
    elif scenario == "S8":
        cfg = WorldConfig(force_present=("ped-0",))
        def sched(e, base):
            pop = _with(base, {"ped-0": "careless", "ped-5": "careless"})
            return pop if e % 5 == 0 else [i for i in pop if i.name != "ped-5"]
        roles = {"frequent": "ped-0", "rare": "ped-5"}
    else:
        raise ValueError(scenario)
    return Lifetime(ss, n_episodes=episodes, config=cfg, schedule=sched), roles


class SensorGaps:
    """Policy-input wrapper (S9): person sightings lose their identity for 2 s every 6 s."""
    PERIOD, GAP = 6.0, 2.0

    def __init__(self, policy):
        self.policy = policy

    def __getattr__(self, item):
        return getattr(self.policy, item)

    def begin_lifetime(self) -> None:
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.policy.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        if (obs.t % self.PERIOD) < self.GAP:
            ents = tuple(dataclasses.replace(e, identity=None, identity_confidence=0.0) if e.kind == "person" else e
                         for e in obs.entities)
            obs = dataclasses.replace(obs, entities=ents)
        self.policy.observe(obs, t)

    def decide(self, t: float) -> int:
        return self.policy.decide(t)

    def end_episode(self, t: float) -> None:
        self.policy.end_episode(t)


def dispositions(lt: Lifetime, episode: int) -> dict[str, str]:
    """Ground truth in force during an episode (evaluation only)."""
    base = lt.population()
    pop = lt.schedule(episode, base) if lt.schedule else base
    out = {i.name: i.disposition for i in base}
    out.update({i.name: i.disposition for i in pop})
    return out
