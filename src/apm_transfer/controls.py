"""Validity controls for adaptation metrics.

`DecayOnlyMemory`: programme I's revising memory that stops updating beliefs from evidence after `freeze_after_s`.
It keeps the evidence-free relaxation toward the type prior. After a world change, any "adaptation" it shows is
produced by forgetting alone. An adaptation metric that credits it is not measuring revision.
"""
from __future__ import annotations

from dataclasses import dataclass

from apm_revise.evidence import Encounter
from apm_revise.memory import RevisionMemory
from apm_revise.policy import Revising


@dataclass
class DecayOnlyMemory(RevisionMemory):
    freeze_after_s: float = float("inf")

    def _end(self, enc: Encounter, t: float) -> None:
        if t < self.freeze_after_s:
            super()._end(enc, t)
            return
        cat = enc.category(self.model.coding)  # logged and counted as usual, but never learned from
        self.counts[enc.g_class()] += 1
        self.log.append((round(t, 2), enc.identity, round(enc.confidence, 3), enc.g_class(), enc.reliable,
                         enc.p_start, cat, None))


class DecayOnly(Revising):
    def __init__(self, *args, freeze_after_s: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_after_s = freeze_after_s

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        self.memory = DecayOnlyMemory(self.type_model, self.lam,  # type: ignore[assignment]  # ObsMemory interface
                                      freeze_after_s=self.freeze_after_s)
