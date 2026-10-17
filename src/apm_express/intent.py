"""ExpressionIntent: an embodiment-independent, bounded description of movement qualities (programme H §1).

Channels are movement qualities from the expressive-motion literature (Laban Effort Time/Space/Flow, approach and
proximity, gaze/orientation, posture), not emotions. Each adapter realises only the channels its embodiment
supports and ignores the rest; values are always clipped to their range, and non-finite values fall back to the
neutral value, so no producer (scripted or learned) can request anything outside the box.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

CHANNELS = ("tempo", "hesitation", "proximity", "directness", "gaze", "scanning", "lean", "head_elevation")
RANGES = {
    "tempo": (0.0, 1.0),  # fraction of time spent progressing (1 = continuous progress)
    "hesitation": (0.0, 1.0),  # irregular pauses / small retreats while approaching
    "proximity": (0.0, 1.0),  # preferred final distance to the target (1 = close)
    "directness": (0.0, 1.0),  # straight path (1) vs indirect, curved approach (0)
    "gaze": (-1.0, 1.0),  # orient toward the target (+1) or away from it (-1)
    "scanning": (0.0, 1.0),  # looking around (head or body sweeps) vs fixed orientation
    "lean": (-1.0, 1.0),  # torso pitch: lean back (-1) .. lean forward (+1), where the body supports it
    "head_elevation": (-1.0, 1.0),  # face down (-1) .. face up (+1), where a head channel is admitted
}
NEUTRAL = {"tempo": 1.0, "hesitation": 0.0, "proximity": 0.5, "directness": 1.0, "gaze": 0.0, "scanning": 0.0,
           "lean": 0.0, "head_elevation": 0.0}


@dataclass(frozen=True)
class ExpressionIntent:
    tempo: float = NEUTRAL["tempo"]
    hesitation: float = NEUTRAL["hesitation"]
    proximity: float = NEUTRAL["proximity"]
    directness: float = NEUTRAL["directness"]
    gaze: float = NEUTRAL["gaze"]
    scanning: float = NEUTRAL["scanning"]
    lean: float = NEUTRAL["lean"]
    head_elevation: float = NEUTRAL["head_elevation"]

    def sanitized(self) -> ExpressionIntent:
        """Every channel finite and inside its range (non-finite -> neutral)."""
        out = {}
        for f in fields(self):
            v = float(getattr(self, f.name))
            lo, hi = RANGES[f.name]
            out[f.name] = NEUTRAL[f.name] if not np.isfinite(v) else min(hi, max(lo, v))
        return ExpressionIntent(**out)

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, c) for c in CHANNELS], dtype=float)

    @classmethod
    def from_array(cls, values) -> ExpressionIntent:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != (len(CHANNELS),):
            raise ValueError(f"expected {len(CHANNELS)} channel values")
        return cls(**dict(zip(CHANNELS, values.tolist(), strict=True))).sanitized()

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


NEUTRAL_INTENT = ExpressionIntent()


def blend(a: ExpressionIntent, b: ExpressionIntent, w: float) -> ExpressionIntent:
    """(1 - w)·a + w·b, sanitized; used for bounded style transitions."""
    w = min(1.0, max(0.0, float(w))) if np.isfinite(w) else 0.0
    return ExpressionIntent.from_array((1 - w) * a.sanitized().as_array() + w * b.sanitized().as_array())
