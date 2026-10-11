"""Transparent baseline appraisal: every frame refers to an explicit predicted or observed outcome.

The appraisal *variables* follow EMA-style appraisal theory (relevance, desirability, likelihood, expectedness,
controllability, uncertainty); the *formulas and constants* below are APM design choices, not an established model.
They are deliberately simple so that later learned appraisers can be compared against them.

  relevance        goal outcomes: priority of the affected goal if active, else INACTIVE_GOAL_RELEVANCE.
                   physical outcomes: 0.6 + 0.4·(1 − m), m = lowest available body margin (strained bodies care more).
  desirability     catalog valence × magnitude (expected magnitude for a prediction).
  likelihood       the hypothesis probability; 1 for an observed outcome.
  expectedness     prediction: agreement with memory, 1 − |p − p_memory|.
                   observation: the probability with which it had been predicted (live hypothesis or memory);
                   an unpredicted outcome is unexpected.
  controllability  physical: stability/control margin × (1 − exp(−horizon/REACTION_S)) for predictions (more time,
                   more control); margin × 0.5 for observed outcomes (coping only). Goal outcomes: GOAL_CONTROL.
  uncertainty      1 − confidence (prediction) or 1 − evidence quality (observation, ×0.7 if unconfirmed),
                   + MISSING_MARGIN_UNCERTAINTY when no body margin is available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from apm.core.outcomes import kind as outcome_kind

INACTIVE_GOAL_RELEVANCE = 0.2
REACTION_S = 1.0
GOAL_CONTROL = 0.6
UNKNOWN_MARGIN = 0.5
MISSING_MARGIN_UNCERTAINTY = 0.2


@dataclass(frozen=True)
class Frame:
    reference_id: str
    reference_type: str          # "hypothesis" | "outcome"
    kind: str
    target: str | None
    relevance: float
    desirability: float
    likelihood: float
    expectedness: float
    controllability: float
    uncertainty: float
    stamp_s: float


def _clip(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, x))


def _relevance(kind_name: str, affected: str, margins: dict[str, float], goal_priority: float | None) -> float:
    k = outcome_kind(kind_name)
    if k.physical:
        m = min(margins.values()) if margins else UNKNOWN_MARGIN
        return _clip(0.6 + 0.4 * (1.0 - m))
    return goal_priority if goal_priority is not None else INACTIVE_GOAL_RELEVANCE


def appraise_hypothesis(h, margins: dict[str, float], goal_priority: float | None, p_memory: float,
                        now_s: float) -> Frame:
    k = outcome_kind(h.kind)
    stability = margins.get("stability_margin", UNKNOWN_MARGIN)
    control = (stability * (1.0 - math.exp(-h.horizon_s / REACTION_S))) if k.physical else GOAL_CONTROL
    uncertainty = 1.0 - h.confidence + (0.0 if margins else MISSING_MARGIN_UNCERTAINTY)
    return Frame(h.hypothesis_id, "hypothesis", h.kind, h.target,
                 relevance=_relevance(h.kind, h.affected, margins, goal_priority),
                 desirability=_clip(k.valence * h.expected_magnitude, -1.0, 1.0),
                 likelihood=h.probability,
                 expectedness=_clip(1.0 - abs(h.probability - p_memory)),
                 controllability=_clip(control), uncertainty=_clip(uncertainty), stamp_s=now_s)


def appraise_outcome(o, margins: dict[str, float], goal_priority: float | None, predicted: float,
                     now_s: float) -> Frame:
    k = outcome_kind(o.kind)
    stability = margins.get("stability_margin", UNKNOWN_MARGIN)
    control = stability * 0.5 if k.physical else GOAL_CONTROL
    quality = o.evidence_quality * (1.0 if o.confirmed else 0.7)
    uncertainty = 1.0 - quality + (0.0 if margins else MISSING_MARGIN_UNCERTAINTY)
    return Frame(o.outcome_id, "outcome", o.kind, o.target,
                 relevance=_relevance(o.kind, o.affected, margins, goal_priority),
                 desirability=_clip(k.valence * o.magnitude, -1.0, 1.0),
                 likelihood=1.0, expectedness=_clip(predicted), controllability=_clip(control),
                 uncertainty=_clip(uncertainty), stamp_s=now_s)
