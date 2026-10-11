"""Programme-G policies composed from G1 regulators and G2 memories."""
from __future__ import annotations

from apm_select.memory import BeliefMemory
from apm_select.regulators import Regulated


class MemoryRegulated(Regulated):
    """B1 rule + regulator (G1) + a belief memory (G2) in place of F's cumulative ObsMemory."""

    def __init__(self, predictor, lam_c: float, lam_s: float, regulator, memory: dict, trace=None, name: str = "M"):
        super().__init__(predictor, lam_c, lam_s, regulator, trace=trace, name=name)
        self.memory_spec = memory

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        self.memory = BeliefMemory(self.memory_spec.get("model", "M0"), self.memory_spec.get("params", {}),
                                   self.memory_spec.get("evidence", "all"))
