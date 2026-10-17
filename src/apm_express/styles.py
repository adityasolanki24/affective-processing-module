"""H1 movement repertoire: designed styles (X1), random styles (X2) and the neutral reference (X0).

Descriptors are experimental labels for motion programmes, not emotions. Channel values follow the cues reported in
the literature review (§1–3):
- slower, interrupted progress and keeping distance for caution and hesitancy;
- direct, uninterrupted, close approach with a forward lean and raised face for assertiveness;
- indirect paths and orienting/scanning for curiosity;
- sustained, smooth, moderate progress for relaxedness;
- continuous fast progress with lively orientation changes for energy.

A channel the embodiment lacks is ignored by its adapter. Whether people read these labels is H2's question.
"""
from __future__ import annotations

import numpy as np

from apm_express.intent import CHANNELS, RANGES, ExpressionIntent

NEUTRAL = ExpressionIntent()  # X0: the validated controller's plain approach

DESIGNED: dict[str, ExpressionIntent] = {  # X1
    "cautious": ExpressionIntent(tempo=0.5, hesitation=0.4, proximity=0.0, directness=0.7, gaze=1.0, scanning=0.3,
                                 lean=-1.0, head_elevation=-0.5),
    "assertive": ExpressionIntent(tempo=1.0, hesitation=0.0, proximity=1.0, directness=1.0, gaze=1.0, scanning=0.0,
                                  lean=1.0, head_elevation=1.0),
    "curious": ExpressionIntent(tempo=0.7, hesitation=0.15, proximity=0.8, directness=0.3, gaze=1.0, scanning=0.9,
                                lean=0.5, head_elevation=0.5),
    "hesitant": ExpressionIntent(tempo=0.6, hesitation=1.0, proximity=0.3, directness=0.9, gaze=-0.5, scanning=0.2,
                                 lean=-0.5, head_elevation=-1.0),
    "relaxed": ExpressionIntent(tempo=0.75, hesitation=0.0, proximity=0.6, directness=0.6, gaze=0.0, scanning=0.15,
                                lean=0.0, head_elevation=0.0),
    "energetic": ExpressionIntent(tempo=1.0, hesitation=0.0, proximity=0.7, directness=0.4, gaze=0.6, scanning=0.6,
                                  lean=0.8, head_elevation=0.8),
}


def random_intent(rng: np.random.Generator) -> ExpressionIntent:
    """X2: every channel uniform over its range (independent draws)."""
    return ExpressionIntent(**{c: float(rng.uniform(*RANGES[c])) for c in CHANNELS})


def with_tempo(intent: ExpressionIntent, tempo: float) -> ExpressionIntent:
    """Speed-matched variant: the same style with only its progress duty cycle changed."""
    values = intent.as_dict()
    values["tempo"] = tempo
    return ExpressionIntent(**values).sanitized()
