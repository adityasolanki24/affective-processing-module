"""J4: experience-dependent memory on NERVA, behind programme H's state-model interface.

**Observations.** `ObsX` extends H's `Obs`. Every field comes from what NERVA measures in an encounter:
- `y`: the self-calibrated push detector (`nerva_world.Evidence.detected`), replacing H's simulator-velocity
  outcome;
- `magnitude`: peak acceleration deviation;
- `category`: the dense evidence code fixed in the J3 addendum.

**TypeStateModel.** Programme I's type belief, with NERVA-specific emissions fitted by EM on J3 data, no labels:
- per identity, a posterior over K types, relaxing toward the prior at rate `lam` per encounter;
- `kind_prior=True`: a new identity starts from the mean posterior of known identities of the same kind, blended with
  π (simple hierarchical pooling, for generalisation to related entities);
- readout: p = b·r, where r_k is type k's *detection* rate, so p means "chance this encounter perturbs me" as in H;
  u = 1 − confidence; g = H's global EMA of detections.

H's fixed lookup L0 maps the readout to an `ExpressionIntent`, unchanged.

`state()` / `load_state()` serialise everything, for the restart-persistence test.

**Lifetimes** (`lifetime_types`). For each history and encounter index, this gives the physical *type* of the entity
met, as the world's hidden manipulation (as in H, now physical types instead of jolt probabilities). Evidence comes
from a bank of physically simulated NERVA encounters of that type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from apm_express.states import BASE, Obs, Probe, Readout, StateModel
from apm_revise.belief import TypeModel

KINDS = ("A", "B", "C")
IDENTITIES = ("A1", "A2", "B1", "B2", "C1", "C2")
NOVEL = ("A3", "B3", "C3")
N_ENCOUNTERS = 48
SAFE, RISKY = "calm", "pushy"

HISTORIES = ("safe", "risky_B", "risky_A", "unpredictable", "change_B", "late_B", "zone_B")


def entity_type(history: str, identity: str, zone: str, k: int, rng: np.random.Generator,
                n: int = N_ENCOUNTERS) -> str:
    kind = identity[0]
    half = k >= n // 2
    if history == "safe":
        return SAFE
    if history == "risky_B":
        return RISKY if kind == "B" else SAFE
    if history == "risky_A":
        return RISKY if kind == "A" else SAFE
    if history == "unpredictable":
        return RISKY if kind == "B" and rng.random() < 0.5 else SAFE
    if history == "change_B":
        return RISKY if kind == "B" and not half else SAFE
    if history == "late_B":
        return RISKY if kind == "B" and half else SAFE
    if history == "zone_B":
        return RISKY if zone == "marked" else SAFE  # the zone, not the entity, is risky: false-attribution test
    raise ValueError(history)


def lifetime_types(history: str, rng: np.random.Generator, n: int = N_ENCOUNTERS) -> list[tuple[str, str, str]]:
    """(identity, zone, physical type) per encounter."""
    out = []
    for k in range(n):
        ident = IDENTITIES[int(rng.integers(len(IDENTITIES)))]
        zone = ("plain", "marked")[int(rng.integers(2))]
        out.append((ident, zone, entity_type(history, ident, zone, k, rng, n)))
    return out


@dataclass(frozen=True)
class ObsX(Obs):
    category: int = 0


class TypeStateModel(StateModel):
    def __init__(self, model: TypeModel, lam: float = 0.0, kind_prior: bool = False, alpha: float = 0.15,
                 name: str = "M_type"):
        self.model, self.lam, self.kind_prior, self.alpha, self.name = model, lam, kind_prior, alpha, name
        self.reset()

    def reset(self) -> None:
        self.b: dict[str, np.ndarray] = {}
        self.k: dict[str, int] = {}     # encounter index of the identity's last update
        self.kinds: dict[str, str] = {}
        self.n = 0                       # encounters so far (the clock)
        self.g = BASE

    def _prior(self, kind: str) -> np.ndarray:
        pi = self.model.pi
        if not self.kind_prior:
            return pi.copy()
        same = [self._at(i) for i, kk in self.kinds.items() if kk == kind]
        if not same:
            return pi.copy()
        return (pi + np.sum(same, axis=0)) / (1 + len(same))

    def _at(self, identity: str) -> np.ndarray:
        pi = self.model.pi
        return pi + (self.b[identity] - pi) * math.exp(-self.lam * (self.n - self.k[identity]))

    def observe(self, ob: Obs) -> None:
        cat = getattr(ob, "category", 0)
        b = self._at(ob.identity) if ob.identity in self.b else self._prior(ob.kind)
        b = b * self.model.E[:, cat]
        self.b[ob.identity] = b / b.sum()
        self.n += 1
        self.k[ob.identity] = self.n
        self.kinds[ob.identity] = ob.kind
        self.g += self.alpha * (ob.y - self.g)

    def readout(self, probe: Probe) -> Readout:
        b = self._at(probe.identity) if probe.identity in self.b else self._prior(probe.kind)
        p = float(b @ self.model.r)
        h = -float(np.sum(b * np.log(np.clip(b, 1e-12, 1.0))))
        u = h / math.log(self.model.K) if self.model.K > 1 else 1.0
        return Readout(p, float(u), self.g)

    def state(self) -> dict:
        return {"b": {k: v.tolist() for k, v in self.b.items()}, "k": dict(self.k), "kinds": dict(self.kinds),
                "n": self.n, "g": self.g}

    def load_state(self, s: dict) -> None:
        self.b = {k: np.asarray(v, float) for k, v in s["b"].items()}
        self.k, self.kinds, self.n, self.g = dict(s["k"]), dict(s["kinds"]), int(s["n"]), float(s["g"])
