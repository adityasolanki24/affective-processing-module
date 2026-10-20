"""Programme I scenario families: G's S1–S9 (unchanged, `apm_select.scenarios`) plus a censoring world.

  S10  one person at a time, risky → safe: exactly one person is present in every episode, always ped-0. ped-0 is
       careless until episode 15, then careful. A robot that keeps away from people it believes risky never sees
       ped-0's behaviour again, so this is the avoidance trap in its strongest form. It is also a boundary test for
       information gathering.
"""
from __future__ import annotations

from apm_research.harness import Lifetime
from apm_select import scenarios as G
from apm_testbeds.wheeled_ctrl import Identity, WorldConfig

EPISODES, SWITCH = G.EPISODES, G.SWITCH
SCENARIOS = (*G.SCENARIOS, "S10")


def make_lifetime(scenario: str, ss, episodes: int = EPISODES) -> tuple[Lifetime, dict]:
    if scenario != "S10":
        return G.make_lifetime(scenario, ss, episodes)
    cfg = WorldConfig(people=(1, 1), force_present=("ped-0",))

    def sched(e: int, base: list[Identity]) -> list[Identity]:
        d = "careless" if e < SWITCH else "careful"
        return [Identity("ped-0", d)]
    return Lifetime(ss, n_episodes=episodes, config=cfg, schedule=sched), {"switched": ["ped-0"], "to": "careful"}


dispositions = G.dispositions
SensorGaps = G.SensorGaps
