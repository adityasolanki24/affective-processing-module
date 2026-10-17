"""J6: belief → expression mappings beyond H's fixed lookup L0 (MASTER_PREREGISTRATION §4.4).

**Supervision available without human judgements: the robot's own consequences.**
- *Time:* from exploration encounters with varied intents, the robot regresses its own arrival time on the intent's
  tempo, hesitation and proximity (`TimeModel`).
- *Perturbation:* the learned NERVA type model already gives, per learned type k and stopping-distance band, the
  probability of strong evidence: an acc8 category ≥ `STRONG` (the top two bins).

Nothing here uses simulator truth or any emotional label as a target.

**Mappings** (all choose from the same small intent grid; other channels as in the neutral style):
- `utility_intent(b)`: argmin over the grid of E[time | intent] + price · Σ_k b_k P(strong | k, band(intent)). This is
  a decision-theoretic expression mapping, with the price as an explicit design value (seconds per strong push);
- `residual_intent(b, l0)`: L0's intent with only its proximity adjusted by δ ∈ {−0.3, …, +0.3}, by the same utility;
- `GruMapping`: a small GRU (numpy inference) over the entity's own evidence history (acc8 category, distance band,
  detected) that outputs a grid intent. It is trained by `j6_train_gru.py` to imitate `utility_intent` from raw
  evidence, i.e. without the explicit type memory, to test whether a recurrent state can replace memory plus
  mapping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from apm_express.adapter import D_FAR, D_NEAR
from apm_express.intent import ExpressionIntent
from apm_express.styles import NEUTRAL
from apm_revise.belief import TypeModel

STRONG = 6
DIST_EDGES = (0.3, 0.5)
PROXIMITY = (0.0, 0.25, 0.5, 0.75, 1.0)
TEMPO = (0.5, 1.0)
HESITATION = (0.0, 0.5)


def grid() -> list[ExpressionIntent]:
    base = NEUTRAL.as_dict()
    return [ExpressionIntent(**{**base, "proximity": p, "tempo": t, "hesitation": h}).sanitized()
            for p in PROXIMITY for t in TEMPO for h in HESITATION]


GRID = grid()


def band_of(proximity: float) -> int:
    return int(np.searchsorted(DIST_EDGES, D_FAR - (D_FAR - D_NEAR) * proximity))


def strong_prob(model: TypeModel, bands: int = 3, C: int = 8) -> np.ndarray:
    """P(strong evidence | type, band) from a band-flattened type model: [K, bands]."""
    E = model.E.reshape(model.K, bands, C)
    return E[:, :, STRONG:].sum(2)


@dataclass
class TimeModel:
    coef: list[float]  # [1, proximity, tempo, hesitation, tempo·hesitation]

    @staticmethod
    def features(it: ExpressionIntent) -> np.ndarray:
        return np.array([1.0, it.proximity, it.tempo, it.hesitation, it.tempo * it.hesitation])

    def predict(self, it: ExpressionIntent) -> float:
        return float(self.features(it) @ np.asarray(self.coef))

    @classmethod
    def fit(cls, intents: list[ExpressionIntent], times: list[float]) -> TimeModel:
        X = np.stack([cls.features(i) for i in intents])
        coef, *_ = np.linalg.lstsq(X, np.asarray(times, float), rcond=None)
        return cls(coef.tolist())


def expected_cost(it: ExpressionIntent, b: np.ndarray, tm: TimeModel, ps: np.ndarray, price: float) -> float:
    return tm.predict(it) + price * float(b @ ps[:, band_of(it.proximity)])


def utility_intent(b: np.ndarray, tm: TimeModel, ps: np.ndarray, price: float) -> tuple[int, ExpressionIntent]:
    costs = [expected_cost(it, b, tm, ps, price) for it in GRID]
    i = int(np.argmin(costs))
    return i, GRID[i]


def residual_intent(b: np.ndarray, l0: ExpressionIntent, tm: TimeModel, ps: np.ndarray,
                    price: float) -> ExpressionIntent:
    best, best_c = l0, math.inf
    base = l0.as_dict()
    for d in (-0.3, -0.15, 0.0, 0.15, 0.3):
        it = ExpressionIntent(**{**base, "proximity": float(np.clip(l0.proximity + d, 0.0, 1.0))}).sanitized()
        c = expected_cost(it, b, tm, ps, price)
        if c < best_c:
            best, best_c = it, c
    return best


def gru_input(category: int | None, band: int, detected: bool) -> np.ndarray:
    x = np.zeros(8 + 3 + 2)
    if category is None:
        x[12] = 1.0  # unobservable encounter
    else:
        x[category] = 1.0
        x[8 + band] = 1.0
        x[11] = float(detected)
    return x


class GruMapping:
    """Numpy GRU inference (weights from j6_train_gru.py, PyTorch GRUCell layout) → grid index."""

    def __init__(self, w: dict):
        self.Wih, self.Whh = np.asarray(w["weight_ih"]), np.asarray(w["weight_hh"])
        self.bih, self.bhh = np.asarray(w["bias_ih"]), np.asarray(w["bias_hh"])
        self.Wo, self.bo = np.asarray(w["out_w"]), np.asarray(w["out_b"])
        self.H = self.Whh.shape[1]

    def init(self) -> np.ndarray:
        return np.zeros(self.H)

    def step(self, h: np.ndarray, x: np.ndarray) -> np.ndarray:
        gi, gh = self.Wih @ x + self.bih, self.Whh @ h + self.bhh
        H = self.H
        r = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
        z = 1 / (1 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        return (1 - z) * n + z * h

    def choose(self, h: np.ndarray) -> int:
        return int(np.argmax(self.Wo @ h + self.bo))
