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
    def __init__(self, world, edges: list[float], stations: dict[str, str], camera=None, video: bool | None = None,
                 approach_from: float | None = None):
        super().__init__(edges, camera)
        self.world, self.stations, self.approach_from = world, stations, approach_from
        self.video = world.video_frames is not None if video is None else video
        self.name = f"{world.body_name}@isaac:{world.scene.name}"
        self.expressive = False
        self.match_motion = False
        self.expression_state = None
        # expression v2: 'memory' (plan from the runtime's learned experience state), 'neutral', or None
        self.program_mode: str | None = None
        self.program_matched = False
        self.experience_source = None  # callable identity -> ExperienceState.readout
        self.program_windows = 3

    def offset(self) -> float:
        from apm_isaac.agents import GAP0, RADIUS
        return RADIUS + self.world.robot.front - GAP0

    last_out: dict = {}

    def _run(self, seed: int, etype: str, d: Decision) -> tuple[dict, dict]:
        """Encounter randomness comes from the world's own per-lifetime stream (paired across systems); `seed`
        is unused here."""
        if self.program_mode is not None:
            return self._run_program(d, etype)
        from apm_transfer.embodied_expression import memory_expression
        intent = memory_expression(d, self.match_motion, self.expression_state) if self.expressive else d.intent()
        if self.match_motion:
            from dataclasses import replace
            intent = replace(intent, tempo=1.0, proximity=0.75, hesitation=0.0)
        self.world.embodied_expression = self.expressive
        # A conservative inspection task for the matched control: every robot
        # observes from the same stand-off, never closer than its safety decision.
        standoff = max(1.0, d.standoff) if self.match_motion else d.standoff
        windows = 3 if self.match_motion else d.windows
        out, truth = self.world.encounter(self.stations[d.identity], etype, intent, windows, standoff,
                                          video=self.video, approach_from=self.approach_from,
                                          return_home=self.approach_from is None)
        self.last_out = out
        from dataclasses import asdict
        out["applied_intent"], out["executed_standoff"] = asdict(intent), standoff
        return out, truth

    def _run_program(self, d: Decision, etype: str) -> tuple[dict, dict]:
        """An encounter with the expressive program. The plan comes from the runtime's experience state for this
        identity (learned from sensing), never from the person's type; the navigation intent comes from the
        plan (matched: fixed speed and stopping distance)."""
        from dataclasses import asdict, replace

        from apm_express.intent import ExpressionIntent
        from apm_transfer.expressive_program import plan
        state = (self.experience_source(d.identity) if self.program_mode == "memory" and self.experience_source
                 else None)
        p = plan(state, matched=self.program_matched)
        intent = replace(ExpressionIntent(), tempo=p.tempo, proximity=p.proximity, hesitation=p.hesitation,
                         directness=p.directness).sanitized()
        self.world.embodied_expression = False
        out, truth = self.world.encounter(self.stations[d.identity], etype, intent, self.program_windows, 0.0,
                                          video=self.video, approach_from=self.approach_from,
                                          return_home=self.approach_from is None, program=p)
        out["applied_intent"], out["executed_standoff"] = asdict(intent), 0.0
        out["experience_readout"] = state
        self.last_out = out
        return out, truth

    def _observation(self, identity: str, kind: str, out: dict, standoff: float) -> Observation:
        physical = out["final_distance"]
        abstract = dict(out, final_distance=round(physical - self.offset(), 3))
        obs = super()._observation(identity, kind, abstract, out.get("executed_standoff", standoff))
        obs.report["d_stop_physical"] = physical
        obs.report["measured_distance"] = out.get("measured_distance")
        for key in ("self_disp_peaks", "standing_peaks", "looming_peaks", "expression", "experience_readout"):
            obs.report[key] = out.get(key)
        reaches, observed = out.get("camera_reaches"), out.get("camera_observed_frames")
        obs.report["camera_reaches"] = ([r for r, n in zip(reaches, observed, strict=True) if n >= 8]
                                        if reaches is not None and observed is not None else reaches)
        return obs
