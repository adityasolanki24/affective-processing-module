"""Embodiment-agnostic outcome semantics (the catalog both OutcomeHypothesis and OutcomeSignal refer to).

A kind states what happened in robot-relevant terms, not how a particular body measured it: a biped's
``stability_loss`` comes from torso tilt, a wheeled robot's from wheel slip or tip-over sensing. Embodiments declare
the kinds they can detect in their EmbodimentProfile. Valence is a design value (sign and default weight of the
consequence for a robot pursuing its goals while staying intact); it is not a claim about emotion.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutcomeKind:
    name: str
    valence: float          # -1..1: adverse < 0 < beneficial
    affects: str            # default affected variable: "self" (body/resources) or "goal"
    physical: bool          # bears on bodily integrity (raises relevance when margins are low)


CATALOG: dict[str, OutcomeKind] = {k.name: k for k in (
    OutcomeKind("collision", -1.0, "self", True),
    OutcomeKind("fall", -1.0, "self", True),
    OutcomeKind("stability_loss", -0.7, "self", True),
    OutcomeKind("control_margin_degraded", -0.4, "self", True),
    OutcomeKind("resource_critical", -0.6, "self", True),
    OutcomeKind("task_interrupted", -0.5, "goal", False),
    OutcomeKind("task_completed", 0.8, "goal", False),
    OutcomeKind("task_progress", 0.3, "goal", False),
    OutcomeKind("recovery", 0.5, "self", True),
    OutcomeKind("benign_contact", 0.4, "self", False),
    OutcomeKind("assistance_received", 0.5, "goal", False),
)}

ADVERSE = frozenset(k for k, v in CATALOG.items() if v.valence < 0)
BENEFICIAL = frozenset(k for k, v in CATALOG.items() if v.valence > 0)


def kind(name: str) -> OutcomeKind:
    try:
        return CATALOG[name]
    except KeyError as error:
        raise KeyError(f"unknown outcome kind {name!r}; extend apm.core.outcomes.CATALOG explicitly") from error
