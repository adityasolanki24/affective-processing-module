"""APM v1.0: the runtime body adapter for Isaac Sim robots.

`apm_transfer.runtime.Runtime` drives any body through `encounter(seed, entity_type, decision)`. Here a decision is
executed in an `IsaacWorld`:
- each identity is a person at their own station;
- the robot approaches through its own expression adapter and safety layers;
- what its sensors saw is coded into the same `Observation` as on the MuJoCo bodies.

**Distance band.** The stopping distance is converted back to the abstract encounter geometry used by the type models
(`agents.GAP0`), so the evidence coding is the one validated in programmes K and L. The physical stopping distance
is kept in `report`.
"""
from __future__ import annotations

from apm_transfer.runtime import Decision, Observation, _Body


class IsaacBody(_Body):
    def __init__(self, world, edges: list[float], stations: dict[str, str], camera=None, video: bool = False):
        super().__init__(edges, camera)
        self.world, self.stations, self.video = world, stations, video
        self.name = f"{world.body_name}@isaac:{world.scene.name}"

    def offset(self) -> float:
        from apm_isaac.agents import GAP0, RADIUS
        return RADIUS + self.world.robot.front - GAP0

    def _run(self, seed: int, etype: str, d: Decision) -> tuple[dict, dict]:
        return self.world.encounter(self.stations[d.identity], etype, d.intent(), d.windows, d.standoff,
                                    video=self.video)

    def _observation(self, identity: str, kind: str, out: dict, standoff: float) -> Observation:
        physical = out["final_distance"]
        abstract = dict(out, final_distance=round(physical - self.offset(), 3))
        obs = super()._observation(identity, kind, abstract, standoff)
        obs.report["d_stop_physical"] = physical
        return obs
