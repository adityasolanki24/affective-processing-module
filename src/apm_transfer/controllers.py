"""Programme L Phase E: simple expression controllers for the fair comparison with the decision-theoretic mapping.

Every controller maps the memory's readout (p: expected perturbation, u: uncertainty, g: global level) to an
`ExpressionIntent`.

- **E0**, H's lookup (`apm_express.states.lookup_intent`). No free parameters; it is E1 with c = 0 and a = 1.
- **E1**, the retuned lookup: caution = clip(a (p − c) + 0.3 (g − 0.05)), then H's relaxed → cautious blend and
  H's uncertainty terms. Free parameters: c, a.
- **E2**, a monotonic rule: proximity falls linearly from 1 at p ≤ lo to 0 at p ≥ lo + width, with tempo 1 and no
  hesitation. Free parameters: lo, width.

E3 and E4 are `infoseek` (greedy and VOI). Their task values are frozen and they have no free parameters.
"""
from __future__ import annotations

import numpy as np

from apm_express.intent import NEUTRAL, ExpressionIntent, blend
from apm_express.states import BASE, CAUTIOUS, RELAXED, W_GLOBAL, Readout


def lookup_tuned(r: Readout, c: float = 0.0, a: float = 1.0) -> ExpressionIntent:
    k = float(np.clip(a * (r.p - c) + W_GLOBAL * (r.g - BASE), 0.0, 1.0))
    it = blend(RELAXED, CAUTIOUS, k).as_dict()
    it["hesitation"] = float(np.clip(it["hesitation"] + 0.6 * r.u, 0.0, 1.0))
    it["scanning"] = float(np.clip(it["scanning"] + 0.5 * r.u, 0.0, 1.0))
    return ExpressionIntent(**it).sanitized()


def monotone(r: Readout, lo: float = 0.2, width: float = 0.2) -> ExpressionIntent:
    prox = float(np.clip(1.0 - (r.p - lo) / max(width, 1e-6), 0.0, 1.0))
    return ExpressionIntent(**{**NEUTRAL, "proximity": prox, "tempo": 1.0, "hesitation": 0.0}).sanitized()


E1_GRID = [{"c": c, "a": a} for c in (0.0, 0.1, 0.2, 0.3) for a in (1.0, 2.0, 4.0)]
E2_GRID = [{"lo": lo, "width": w} for lo in (0.1, 0.2, 0.3) for w in (0.1, 0.2, 0.3, 0.4)]
