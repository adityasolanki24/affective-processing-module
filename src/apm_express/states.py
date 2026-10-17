"""Internal-state models for H3–H6 and the fixed expression lookup L0. Numpy only.

Every model observes only what the robot measures after each encounter:
- identity tag, kind, zone;
- the observed outcome `y` (perturbed or not: measured peak base speed ≥ 0.04 m/s, H3/H4 addendum);
- `magnitude` (measured peak base speed / 0.1 m/s).

It never sees the world's hidden jolt probability or history label, except `OracleLookup`, which is the explicitly
labelled human-designed history-to-style control.

A model answers `readout(probe)` with a `Readout` of three values in [0, 1]:
- `p`: expected chance that this encounter perturbs the robot;
- `u`: uncertainty;
- `g`: a global, non-entity state such as "recent experience overall".

The **fixed, transparent lookup L0** maps a readout to an `ExpressionIntent`. It is shared by every model, so
differences in expression come from the state, not from the mapping. The PAD model is the exception: as in
APM v0.1 / NERVA, PAD drives expression through its own fixed mapping (`pad_intent`).

Models (H3 names):
- S0 `Constant`: no state, current observation only.
- S1 `FixedStyle`.
- S2 `ScalarEMA`.
- S3 `SmallState`: global risk, volatility, identity novelty.
- S4 `PADState`.
- `EntityMemory`: the A0 explicit memory, per-identity Beta with a kind-level pool prior plus a global EMA risk
  filter; optionally keyed by zone (`context=True`) or with change-point beliefs (`changepoint=True`).
- The learned recurrent models S5–S7 live in `states_learned.py` (torch).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from apm_express.intent import ExpressionIntent, blend
from apm_express.styles import DESIGNED
from apm_select.memory import ChangePoint, Cumulative

BASE = 0.05  # prior expectation of being perturbed by an unfamiliar entity
MAG_SCALE = 1.0  # magnitude normalisation for PAD dominance (magnitude = peak speed / 0.1 m/s)


@dataclass(frozen=True)
class Obs:
    identity: str
    kind: str
    zone: str
    y: float  # observed: 1 if the encounter perturbed the robot (measured disturbance above threshold)
    magnitude: float  # observed disturbance size (peak base speed / 0.1 m/s)


@dataclass(frozen=True)
class Probe:
    identity: str
    kind: str
    zone: str


@dataclass(frozen=True)
class Readout:
    p: float
    u: float
    g: float


class StateModel:
    name = "base"
    fixed_intent: ExpressionIntent | None = None

    def reset(self) -> None: ...

    def observe(self, ob: Obs) -> None: ...

    def readout(self, probe: Probe) -> Readout:
        return Readout(BASE, 0.5, BASE)


class Constant(StateModel):
    """S0: no internal state; expression from the present observation only (which carries no risk cue)."""
    name = "S0_none"


class FixedStyle(StateModel):
    """S1: one fixed, manually selected style regardless of experience."""
    name = "S1_fixed"

    def __init__(self, style: str = "relaxed"):
        self.fixed_intent = DESIGNED[style]


@dataclass
class ScalarEMA(StateModel):
    """S2: one scalar, the exponentially smoothed observed outcome."""
    alpha: float = 0.15
    name: str = "S2_scalar"
    g: float = BASE

    def reset(self) -> None:
        self.g = BASE

    def observe(self, ob: Obs) -> None:
        self.g += self.alpha * (ob.y - self.g)

    def readout(self, probe: Probe) -> Readout:
        return Readout(self.g, 0.0, self.g)


@dataclass
class SmallState(StateModel):
    """S3: (global risk EMA, volatility EMA, novelty of the probed identity): a small continuous state."""
    alpha: float = 0.15
    name: str = "S3_small"
    g: float = BASE
    vol: float = 0.0
    counts: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.g, self.vol, self.counts = BASE, 0.0, {}

    def observe(self, ob: Obs) -> None:
        err = ob.y - self.g
        self.g += self.alpha * err
        self.vol += self.alpha * (err * err - self.vol)
        self.counts[ob.identity] = self.counts.get(ob.identity, 0) + 1

    def readout(self, probe: Probe) -> Readout:
        novelty = 1.0 / (1.0 + self.counts.get(probe.identity, 0))
        u = float(np.clip(2.0 * math.sqrt(max(self.vol, 0.0)) * 0.5 + 0.5 * novelty, 0.0, 1.0))
        return Readout(self.g, u, self.g)


@dataclass
class PADState(StateModel):
    """S4: a global pleasure–arousal–dominance state updated by appraising each observed outcome (APM v0.1 style):
    pleasure falls with perturbation, arousal rises with perturbation and surprise, dominance (coping) falls with
    the size of the perturbation. Values in [-1, 1]; decays toward a calm baseline between events."""
    alpha: float = 0.15
    name: str = "S4_pad"
    P: float = 0.5
    A: float = -0.5
    D: float = 0.5
    expect: float = BASE

    def reset(self) -> None:
        self.P, self.A, self.D, self.expect = 0.5, -0.5, 0.5, BASE

    def observe(self, ob: Obs) -> None:
        surprise = abs(ob.y - self.expect)
        size = min(1.0, ob.magnitude / MAG_SCALE)
        self.P += self.alpha * ((1.0 - 2.0 * ob.y) - self.P)
        self.A += self.alpha * ((2.0 * max(ob.y, surprise) - 1.0) - self.A)
        self.D += self.alpha * ((1.0 - 2.0 * size) - self.D)
        self.expect += self.alpha * (ob.y - self.expect)

    def readout(self, probe: Probe) -> Readout:
        return Readout((1.0 - self.P) / 2.0, (self.A + 1.0) / 2.0, (1.0 - self.D) / 2.0)


class EntityMemory(StateModel):
    """A0's explicit memory: per-identity Beta beliefs (programme F/G `Cumulative`, prior strength 2) with the kind
    pool as prior, and a global EMA risk filter. context=True keys beliefs by (identity, zone) and pools by
    (kind, zone); changepoint=True uses G2's BOCPD belief (hazard 0.05) for identities."""

    def __init__(self, context: bool = False, changepoint: bool = False, alpha: float = 0.15):
        self.context, self.changepoint, self.alpha = context, changepoint, alpha
        self.name = "M_" + ("ctx" if context else "id") + ("_cp" if changepoint else "")
        self.reset()

    def reset(self) -> None:
        self.ids: dict = {}
        self.pools: dict = {}
        self.g = BASE

    def _keys(self, identity: str, kind: str, zone: str):
        return ((identity, zone), (kind, zone)) if self.context else (identity, kind)

    def observe(self, ob: Obs) -> None:
        ik, pk = self._keys(ob.identity, ob.kind, ob.zone)
        pool = self.pools.setdefault(pk, Cumulative())
        prior = pool.p(BASE)[0]
        belief = self.ids.setdefault(ik, ChangePoint(0.05) if self.changepoint else Cumulative())
        belief.update(ob.y, 1.0, prior)
        pool.update(ob.y, 1.0, BASE)
        self.g += self.alpha * (ob.y - self.g)

    def readout(self, probe: Probe) -> Readout:
        ik, pk = self._keys(probe.identity, probe.kind, probe.zone)
        pool = self.pools.get(pk)
        prior = pool.p(BASE)[0] if pool else BASE
        belief = self.ids.get(ik)
        if belief is None:
            conf_pool = pool.p(BASE)[1] if pool else 0.0
            return Readout(prior, 1.0 - 0.5 * conf_pool, self.g)
        p, conf = belief.p(prior)
        return Readout(float(p), float(1.0 - conf), self.g)


@dataclass
class OracleLookup(StateModel):
    """Control: a human-designed history-to-style lookup that is *told* the history condition (not learned, not
    from experience). It shows what hand-authoring achieves when the designer knows the world."""
    history: str = "safe"
    name: str = "oracle_lookup"

    def readout(self, probe: Probe) -> Readout:
        from apm_express.develop import N_ENCOUNTERS, jolt_probability
        return Readout(jolt_probability(self.history, probe.identity, probe.zone, N_ENCOUNTERS - 1), 0.0, BASE)


# ── the fixed expression lookup L0 ────────────────────────────────────────────
W_GLOBAL = 0.3
RELAXED, CAUTIOUS = DESIGNED["relaxed"], DESIGNED["cautious"]


def caution(r: Readout) -> float:
    return float(np.clip(r.p + W_GLOBAL * (r.g - BASE), 0.0, 1.0))


def lookup_intent(r: Readout) -> ExpressionIntent:
    """L0: caution blends relaxed -> cautious; uncertainty adds hesitation and scanning."""
    k = caution(r)
    it = blend(RELAXED, CAUTIOUS, k).as_dict()
    it["hesitation"] = float(np.clip(it["hesitation"] + 0.6 * r.u, 0.0, 1.0))
    it["scanning"] = float(np.clip(it["scanning"] + 0.5 * r.u, 0.0, 1.0))
    return ExpressionIntent(**it).sanitized()


def pad_intent(m: PADState) -> ExpressionIntent:
    """S4's own fixed mapping, in the directions NERVA's pad_style used: low pleasure/dominance -> withdrawn,
    cautious, lean back and face down; high arousal -> more hesitation and scanning (vigilance)."""
    k = float(np.clip((1.0 - m.P) / 2.0 * 0.6 + (1.0 - m.D) / 2.0 * 0.4, 0.0, 1.0))
    it = blend(RELAXED, CAUTIOUS, k).as_dict()
    a = (m.A + 1.0) / 2.0
    it["hesitation"] = float(np.clip(it["hesitation"] + 0.6 * a, 0.0, 1.0))
    it["scanning"] = float(np.clip(it["scanning"] + 0.5 * a, 0.0, 1.0))
    return ExpressionIntent(**it).sanitized()


def intent_of(model: StateModel, probe: Probe) -> ExpressionIntent:
    if model.fixed_intent is not None:
        return model.fixed_intent
    if isinstance(model, PADState):
        return pad_intent(model)
    return lookup_intent(model.readout(probe))
