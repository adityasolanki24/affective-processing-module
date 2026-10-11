"""Programme-G policies composed from G1 regulators and G2 memories."""
from __future__ import annotations

import numpy as np

from apm_select.memory import RELIABLE_CONF, BeliefMemory
from apm_select.regulators import Regulated
from apm_testbeds.wheeled_ctrl import DETOUR, SLOW, YIELD

PROBE_NEAR_M, PROBE_ALONE_M = 1.5, 2.0


class MemoryRegulated(Regulated):
    """B1 rule + regulator (G1) + a belief memory (G2) in place of F's cumulative ObsMemory.

    Optional information gathering (`memory["probe"]`): when the rule would avoid (YIELD/DETOUR) the nearest person,
    that person is within 1.5 m, nobody else is within 2.0 m, and the per-episode probe budget is not used up, the
    robot approaches at SLOW instead —
      mode "uncertainty": only if the person's identity is reliable and memory confidence < conf_max;
      mode "random":      with probability p_random (budget-matched control; no use of uncertainty).
    The safety supervisor is unchanged; no other action is altered.
    """

    def __init__(self, predictor, lam_c: float, lam_s: float, regulator, memory: dict, trace=None, name: str = "M"):
        super().__init__(predictor, lam_c, lam_s, regulator, trace=trace, name=name)
        self.memory_spec = memory
        self.probe = memory.get("probe")
        self.probes = 0

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        spec = self.memory_spec
        self.memory = BeliefMemory(spec.get("model", "M0"), spec.get("params", {}), spec.get("evidence", "all"))
        self.probes = 0
        self.probe_log: list[tuple[int, int, int]] = []  # (episode probe index, rule action, executed action)
        self.rng = np.random.default_rng((self.probe or {}).get("seed", 0))

    def begin_episode(self, t0: float) -> None:
        super().begin_episode(t0)
        self.probes_episode = 0

    def decide(self, t: float) -> int:
        a = super().decide(t)
        pr = self.probe
        if not pr or a not in (YIELD, DETOUR) or self.probes_episode >= pr["budget"]:
            return a
        obs = self.state.last_obs
        people = sorted((e for e in obs.entities if e.kind == "person"), key=lambda e: e.distance) if obs else []
        if not people or people[0].distance > PROBE_NEAR_M or any(e.distance < PROBE_ALONE_M for e in people[1:]):
            return a
        e = people[0]
        if pr["mode"] == "uncertainty":
            if e.identity is None or e.identity_confidence < RELIABLE_CONF:
                return a
            if self.memory.p_collision(e.identity, "person")[1] >= pr["conf_max"]:
                return a
        elif self.rng.random() >= pr["p_random"]:
            return a
        self.probes += 1
        self.probes_episode += 1
        self.probe_log.append((self.probes_episode, a, SLOW))
        return SLOW
