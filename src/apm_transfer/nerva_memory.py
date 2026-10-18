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
                 name: str = "M_type", transition_each_meeting: bool = False, log_space: bool = False):
        self.model, self.lam, self.kind_prior, self.alpha, self.name = model, lam, kind_prior, alpha, name
        # Programme K deviation D1. The change model must apply one transition per encounter *including* the one being
        # entered (predict, then update). Without it, an identity met in consecutive encounters never relaxes, and a
        # few decisive observations drive the belief to exactly 0 or 1, from which no evidence can revise it.
        # False reproduces programmes J and K up to K3-X / K5-X exactly.
        self.transition_each_meeting = transition_each_meeting
        # Programme L audit A6: accumulate the evidence of one encounter in log space and normalise once (no
        # intermediate underflow). False reproduces K's sequential product exactly.
        self.log_space = log_space
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
        gap = self.n - self.k[identity] + (1 if self.transition_each_meeting else 0)
        return pi + (self.b[identity] - pi) * math.exp(-self.lam * gap)

    def observe(self, ob: Obs) -> None:
        cat = getattr(ob, "category", 0)
        b = self._at(ob.identity) if ob.identity in self.b else self._prior(ob.kind)
        b = b * self.model.E[:, cat]
        self.b[ob.identity] = b / b.sum()
        self.n += 1
        self.k[ob.identity] = self.n
        self.kinds[ob.identity] = ob.kind
        self.g += self.alpha * (ob.y - self.g)

    def observe_many(self, identity: str, kind: str, categories: list[int], y: float) -> None:
        """Programme K: several conditionally independent observations from one encounter (sensing windows, camera
        codes), as flattened emission columns. The clock advances once. With no categories (nothing observable),
        nothing is learned and the clock still advances."""
        b = self._at(identity) if identity in self.b else self._prior(kind)
        if self.log_space:
            with np.errstate(divide="ignore"):
                lp = np.log(b) + sum((np.log(self.model.E[:, c]) for c in categories), np.zeros_like(b))
            if np.isfinite(lp.max()):  # all types impossible (zero emissions): keep the predicted belief
                lp -= lp.max()
                b = np.exp(lp)
                b = b / b.sum()
        else:
            for cat in categories:
                b = b * self.model.E[:, cat]
                b = b / b.sum()
        self.n += 1
        if categories:
            self.b[identity] = b
            self.k[identity] = self.n
            self.kinds[identity] = kind
            self.g += self.alpha * (y - self.g)

    def belief(self, identity: str, kind: str) -> np.ndarray:
        return self._at(identity) if identity in self.b else self._prior(kind)

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


class GruMemory(StateModel):
    """J4's recurrent baseline: per identity, a small GRU state over that identity's own evidence (acc8 category,
    detected). Readout p = σ(w·h + b), the predicted chance that the next meeting perturbs the robot. It is trained
    self-supervised on V lifetimes (`j4_gru.py`: predict the identity's next detection), with no type labels, no type
    model and no kind pooling. A new identity starts from h = 0. u = 1 / (1 + meetings); g = H's global EMA."""

    name = "gru"

    def __init__(self, weights: dict):
        from apm_transfer.expression_map import GruMapping
        self.net = GruMapping(weights)
        self.reset()

    def reset(self) -> None:
        self.h: dict[str, np.ndarray] = {}
        self.n: dict[str, int] = {}
        self.g = BASE

    def observe(self, ob: Obs) -> None:
        from apm_transfer.expression_map import gru_input
        cat = int(getattr(ob, "category", 0)) % 8
        h = self.h.get(ob.identity, self.net.init())
        self.h[ob.identity] = self.net.step(h, gru_input(cat, 1, bool(ob.y)))
        self.n[ob.identity] = self.n.get(ob.identity, 0) + 1
        self.g += 0.15 * (ob.y - self.g)

    def readout(self, probe: Probe) -> Readout:
        h = self.h.get(probe.identity, self.net.init())
        z = float((self.net.Wo @ h + self.net.bo)[0])
        return Readout(1.0 / (1.0 + math.exp(-z)), 1.0 / (1 + self.n.get(probe.identity, 0)), self.g)


class PriorOnlyMemory:
    """Programme L Phase D baseline: no memory. Every belief is the type prior π."""

    name = "M0_prior"

    def __init__(self, model: TypeModel):
        self.model = model

    def belief(self, identity: str, kind: str) -> np.ndarray:
        return self.model.pi.copy()

    def observe_many(self, identity: str, kind: str, categories: list[int], y: float) -> None:
        return None


class LeakyEvidenceMemory:
    """Programme L Phase D alternative: a recency-weighted risk estimate.

    Per identity, it keeps a leaky sum of per-type log-likelihoods: s <- (1 - alpha) s + sum_c log E[:, c], updated
    at each of *that identity's* meetings that has observations. The belief is softmax(log π + s). Old evidence
    fades geometrically (half-life ln 0.5 / ln(1 - alpha) meetings), whatever happened in between. It has no change
    model and no global clock."""

    name = "M2_leaky"

    def __init__(self, model: TypeModel, alpha: float = 0.2):
        self.model, self.alpha = model, alpha
        self.s: dict[str, np.ndarray] = {}
        self.logE = np.log(np.clip(model.E, 1e-300, None))

    def belief(self, identity: str, kind: str) -> np.ndarray:
        lp = np.log(self.model.pi) + self.s.get(identity, 0.0)
        lp = lp - lp.max()
        b = np.exp(lp)
        return b / b.sum()

    def observe_many(self, identity: str, kind: str, categories: list[int], y: float) -> None:
        if not categories:
            return
        ll = self.logE[:, categories].sum(1)
        self.s[identity] = (1 - self.alpha) * self.s.get(identity, np.zeros_like(ll)) + (ll - ll.max())


class ContextKeyedMemory:
    """Programme L Phase D alternative: the Bayesian type memory kept separately per (identity, context). The
    context is something the robot knows itself, e.g. the zone it is approaching."""

    name = "M3_context"

    def __init__(self, inner: TypeStateModel):
        self.inner = inner
        self.context = ""

    def key(self, identity: str) -> str:
        return f"{identity}@{self.context}"

    def belief(self, identity: str, kind: str) -> np.ndarray:
        return self.inner.belief(self.key(identity), kind)

    def observe_many(self, identity: str, kind: str, categories: list[int], y: float) -> None:
        self.inner.observe_many(self.key(identity), kind, categories, y)
