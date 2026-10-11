"""Explicit outcome-grounded memory for decision systems: the APM `EntityMemory` class driven directly by `Obs`.

The lifecycle mirrors `apm.core.runtime.APM` exactly (tests/test_wheeled_ctrl.py checks equality on the same stream):
every sighting calls `track`; a collision outcome is attributed to its contact target; an encounter ends when its
track has not been seen for more than `ENTITY_TTL_S` (the APM context's entity TTL), evaluated at every step.
Untargeted outcomes (slip) are never attributed to an entity.
"""
from __future__ import annotations

from apm.adapters.wheeled_ctrl import OUTCOMES
from apm.memory.entities import EntityMemory

ENTITY_TTL_S = 1.5


class ObsMemory:
    def __init__(self) -> None:
        self.mem = EntityMemory(OUTCOMES)
        self.last_seen: dict[str, float] = {}

    def observe(self, obs, t: float) -> None:
        for e in obs.entities:
            self.mem.track(e.track, e.identity, e.identity_confidence if e.identity else 0.0, e.kind, t)
            self.last_seen[e.track] = t
        for event in obs.events:
            if event[0] == "collision":
                self.mem.outcome(event[1], "collision", 1.0)
        self.expire(t)

    def expire(self, t: float) -> None:
        for track in [k for k, s in self.last_seen.items() if t - s > ENTITY_TTL_S]:
            self.mem.end(track, t)
            del self.last_seen[track]

    def p_collision(self, identity: str | None, kind: str) -> tuple[float, float]:
        return self.mem.p("collision", identity, kind)
