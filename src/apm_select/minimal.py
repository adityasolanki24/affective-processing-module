"""G8 minimisation components for the A0 family.

* `HeuristicPredictor` — a transparent, hand-written consequence model replacing the learned MLP ensemble (tests
  whether *neural* prediction is needed). For each action it scores, from the same decision features:
    P_col(a)  = 1 − Π_people (1 − c_a · ttc_max_i · m_i),  m_i = 0.5 + memory P(collision | identity)
    P_slip(a) = s_a · (1 − traction margin)
    progress(a) = v_a · max(0, cos(goal bearing)) · min(1, goal distance / 2 m) · 2 s
  with c = (1.0, 0.45, 0.1, 0.7), s = (1.0, 0.4, 0.05, 0.8), v = (0.8, 0.3, 0.0, 0.6) m/s for GO/SLOW/YIELD/DETOUR
  (the testbed's speed caps). Constants are fixed here, before any G8 data; λ is tuned with the same budget as G1.
* `SingleMember` — one member of programme F's 5-member ensemble (tests whether ensembling/averaging matters).
* `NoMemory` — the B1 rule with the explicit-memory feature columns zeroed (tests whether entity memory matters).
"""
from __future__ import annotations

import numpy as np

from apm_research.features import K_PEOPLE, MEM_CONF, MEM_P, PERSON_W
from apm_research.predictor import EnsemblePredictor
from apm_select.regulators import Regulated
from apm_testbeds.wheeled_ctrl import N_ACTIONS

C_COL = np.array([1.0, 0.45, 0.1, 0.7])
C_SLIP = np.array([1.0, 0.4, 0.05, 0.8])
SPEED = np.array([0.8, 0.3, 0.0, 0.6])
HORIZON_S = 2.0


class HeuristicPredictor:
    parameters = 0

    def predict_actions(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, float)
        hazard = []
        for i in range(K_PEOPLE):
            b = 7 + i * PERSON_W
            if x[b] > 0:  # present
                hazard.append(x[b + 5] * (0.5 + x[b + 6]))  # ttc_max × (0.5 + memory p)
        p_col = np.array([1.0 - np.prod([1.0 - min(1.0, c * h) for h in hazard]) if hazard else 0.0 for c in C_COL])
        p_slip = C_SLIP * (1.0 - x[2])
        cosb = x[4]
        dist = x[3] * 8.5
        progress = SPEED * max(0.0, cosb) * min(1.0, dist / 2.0) * HORIZON_S
        return p_col, p_slip, progress


class SingleMember(EnsemblePredictor):
    @classmethod
    def first_of(cls, ens: EnsemblePredictor) -> SingleMember:
        return cls(ens.members[:1])


class NoMemory(Regulated):
    """A0 with the explicit-memory feature columns zeroed (the memory still runs but is never used)."""

    def features(self, t: float) -> np.ndarray:
        x = super().features(t)
        x[MEM_P], x[MEM_CONF] = 0.0, 0.0
        self.last_features = x
        return x


assert N_ACTIONS == len(C_COL)
