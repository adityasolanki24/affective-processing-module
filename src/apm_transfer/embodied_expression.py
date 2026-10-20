"""Designed movement mapping from learned expectations, without emotion labels.

The Bayesian memory is learned from robot observations. This mapping is designed,
not a learned expressive policy. It consumes only the runtime decision's learned
risk estimate and uncertainty, never the simulator's entity type.
"""
from dataclasses import replace

import numpy as np

from apm_express.intent import ExpressionIntent
from apm_transfer.runtime import Decision


def memory_expression(decision: Decision, match_motion: bool = False,
                      sensor_state: dict | None = None) -> ExpressionIntent:
    p, u = float(decision.p_risk), float(decision.uncertainty)
    if not np.isfinite(p) or not np.isfinite(u):
        return decision.intent()
    p, u = np.clip([p, u], 0., 1.)
    # Inspect uncertain surroundings; attend more steadily to familiar targets.
    # Risk also changes inspection and path directness, not just final distance.
    inspection = 0.6 * u + 0.4 * p
    if sensor_state is not None:
        # Pilot alternative: learned frequency of camera-observed movement.
        # A designed monotone mapping, not a learned motion policy or emotion claim.
        p = float(np.clip(sensor_state["movement_probability"], 0., 1.))
        u = float(np.clip(sensor_state["uncertainty"], 0., 1.))
        inspection = float(np.clip(2.0 * p, 0., 1.))
    intent = replace(decision.intent(), scanning=float(0.10 + 0.80 * inspection),
                     gaze=1.0, directness=float(1.0 - 0.45 * inspection),
                     head_elevation=float(0.4 * (1.0 - u) - 0.35 * p))
    if match_motion:
        intent = replace(intent, tempo=1.0, proximity=0.75, hesitation=0.0)
    return intent.sanitized()
