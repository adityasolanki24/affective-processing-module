"""G2 belief memories: per-identity collision risk that can be revised, with explicit selective-feedback handling.

Drop-in replacement for `apm_research.memory.ObsMemory` (same `observe`, `expire`, `p_collision` interface), driven
only by `Obs`. An *encounter* is a person track's lifetime (ends when unseen for more than ENTITY_TTL_S, as in F).
Each ended encounter is classified from observations only:

  collision   a collision event with this track occurred
  exposed     no collision, but the person came within EXPOSED_M while the robot moved faster than MOVING_MPS
              (a collision was physically possible)
  avoided     neither, and the robot had chosen SLOW/YIELD/DETOUR while the person was within AVOID_M
  unexposed   none of the above (never close while moving)
  unreliable  reported identity missing or with confidence below RELIABLE_CONF (any class); evidence for the
              type pool only

Evidence models: "all" (every ended encounter is a Bernoulli trial: collision = 1, otherwise 0 — F's EntityMemory)
and "exposed" (only collision and exposed encounters are trials; avoided/unexposed encounters teach nothing). No
counterfactual labels are ever created.

Belief models over each identity's Bernoulli trials (weights = identity confidence), with a type-level pool used as
the prior for individuals and for unknown people:
  M0 cumulative Beta (F's EntityMemory rule: prior strength 2 around the pool mean)
  M1 exponential forgetting (counts × γ before each new trial)
  M2 sliding window (last W trials)
  M3 hierarchical Beta: prior strength estimated from between-identity dispersion (empirical Bayes), no forgetting
  M4 Bayesian online change-point detection (Adams & MacKay 2007), constant hazard h, Beta prior from the pool,
     run length truncated at RMAX
  M5 two-state hidden Markov filter (low/high risk, emission q_low/q_high, switch probability s per trial)
  M6 learned recurrent per-identity memory: a GRU cell (hidden 8) over the identity's own trial sequence
     (input [y, w]) with a logistic read-out mixed with the pool prior; weights trained offline on G2-T encounter
     logs (scripts/g2_m6_train.py), pure-NumPy inference
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from apm_testbeds.wheeled_ctrl import DETOUR, SLOW, YIELD

ENTITY_TTL_S = 1.5
EXPOSED_M, MOVING_MPS, AVOID_M, RELIABLE_CONF = 0.5, 0.15, 1.5, 0.5
BASE_RATE, PRIOR_STRENGTH = 0.05, 2.0
CLASSES = ("collision", "exposed", "avoided", "unexposed")


@dataclass
class Encounter:
    track: str
    identity: str | None = None
    confidence: float = 0.0
    start: float = 0.0
    collision: bool = False
    exposed: bool = False
    avoided: bool = False

    def klass(self) -> str:
        return "collision" if self.collision else "exposed" if self.exposed else \
            "avoided" if self.avoided else "unexposed"

    @property
    def reliable(self) -> bool:
        return self.identity is not None and self.confidence >= RELIABLE_CONF


class Belief:
    """One identity's (or the pool's) belief about its per-trial collision probability."""
    def update(self, y: float, w: float, prior: float) -> None: ...
    def p(self, prior: float) -> tuple[float, float]:
        return prior, 0.0


class Cumulative(Belief):
    def __init__(self, gamma: float = 1.0, strength: float = PRIOR_STRENGTH):
        self.n = self.k = 0.0
        self.gamma, self.strength = gamma, strength

    def update(self, y: float, w: float, prior: float) -> None:
        self.n, self.k = self.gamma * self.n + w, self.gamma * self.k + w * y

    def p(self, prior: float) -> tuple[float, float]:
        s = self.strength
        return (self.k + s * prior) / (self.n + s), self.n / (self.n + s)


class TimeDecay(Cumulative):
    """M1t: evidence decays with lifetime time, n, k ← n·e^(−Δt/τ), k·e^(−Δt/τ), whether or not new evidence arrives,
    so an avoided identity drifts back toward the pool prior (G2b). `clock()` returns the memory's current time."""

    def __init__(self, tau_s: float, clock):
        super().__init__()
        self.tau, self.clock, self.t = tau_s, clock, None

    def _decay(self) -> None:
        now = self.clock()
        if self.t is not None and now > self.t:
            f = math.exp(-(now - self.t) / self.tau)
            self.n, self.k = self.n * f, self.k * f
        self.t = now

    def update(self, y: float, w: float, prior: float) -> None:
        self._decay()
        super().update(y, w, prior)

    def p(self, prior: float) -> tuple[float, float]:
        self._decay()
        return super().p(prior)


class Window(Belief):
    def __init__(self, size: int):
        self.trials: deque = deque(maxlen=max(1, int(size)))

    def update(self, y: float, w: float, prior: float) -> None:
        self.trials.append((y, w))

    def p(self, prior: float) -> tuple[float, float]:
        n = sum(w for _, w in self.trials)
        k = sum(w * y for y, w in self.trials)
        return (k + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH), n / (n + PRIOR_STRENGTH)


class ChangePoint(Belief):
    """BOCPD over Bernoulli trials; each run length r keeps Beta(a_r, b_r) with the pool prior at r = 0."""
    RMAX = 60

    def __init__(self, hazard: float, strength: float = PRIOR_STRENGTH):
        self.h, self.s = hazard, strength
        self.r = np.array([1.0])          # P(run length)
        self.k = np.array([0.0])          # weighted successes per run length (beyond the prior)
        self.n = np.array([0.0])

    def _pred(self, prior: float) -> np.ndarray:
        return (self.k + self.s * prior) / (self.n + self.s)

    def update(self, y: float, w: float, prior: float) -> None:
        pr = self._pred(prior)
        lik = (pr if y >= 0.5 else 1.0 - pr) ** w
        growth = self.r * lik * (1.0 - self.h)
        cp = float(np.sum(self.r * lik * self.h))
        r = np.concatenate([[cp], growth])
        k = np.concatenate([[0.0], self.k]) + w * y * np.concatenate([[0.0], np.ones_like(self.k)])
        n = np.concatenate([[0.0], self.n]) + w * np.concatenate([[0.0], np.ones_like(self.n)])
        # a change point before this trial: the new run starts with this trial as its first observation
        k[0], n[0] = w * y, w
        if len(r) > self.RMAX:
            r[self.RMAX - 1] += r[self.RMAX:].sum()
            r, k, n = r[:self.RMAX], k[:self.RMAX], n[:self.RMAX]
        total = r.sum()
        self.r = r / total if total > 0 else np.full_like(r, 1.0 / len(r))
        self.k, self.n = k, n

    def p(self, prior: float) -> tuple[float, float]:
        p = float(self.r @ self._pred(prior))
        n_eff = float(self.r @ self.n)
        return p, n_eff / (n_eff + self.s)


class HiddenState(Belief):
    """Two-state forward filter: P(high-risk state) with switch probability s per trial."""

    def __init__(self, q_low: float, q_high: float, switch: float):
        self.q = (q_low, q_high)
        self.s = switch
        self.p_high: float | None = None
        self.n = 0.0

    def _init(self, prior: float) -> None:
        if self.p_high is None:
            lo, hi = self.q
            self.p_high = min(1.0, max(0.0, (prior - lo) / max(1e-6, hi - lo)))

    def update(self, y: float, w: float, prior: float) -> None:
        self._init(prior)
        ph = self.p_high * (1 - self.s) + (1 - self.p_high) * self.s
        lo, hi = self.q
        l_hi = (hi if y >= 0.5 else 1 - hi) ** w
        l_lo = (lo if y >= 0.5 else 1 - lo) ** w
        self.p_high = ph * l_hi / max(1e-300, ph * l_hi + (1 - ph) * l_lo)
        self.n += w

    def p(self, prior: float) -> tuple[float, float]:
        self._init(prior)
        ph = self.p_high * (1 - self.s) + (1 - self.p_high) * self.s
        lo, hi = self.q
        return ph * hi + (1 - ph) * lo, self.n / (self.n + PRIOR_STRENGTH)


class Neural(Belief):
    """GRU over trials; p = σ(w·h + b + logit(prior)) so that an empty history returns the prior."""

    def __init__(self, weights: dict):
        self.w = {k: np.asarray(v, float) for k, v in weights.items()}
        self.h = np.zeros(self.w["W_hr"].shape[0])
        self.n = 0.0

    def update(self, y: float, w: float, prior: float) -> None:
        x = np.array([y, w])
        W = self.w
        r = 1 / (1 + np.exp(-(W["W_ir"] @ x + W["b_ir"] + W["W_hr"] @ self.h + W["b_hr"])))
        z = 1 / (1 + np.exp(-(W["W_iz"] @ x + W["b_iz"] + W["W_hz"] @ self.h + W["b_hz"])))
        n = np.tanh(W["W_in"] @ x + W["b_in"] + r * (W["W_hn"] @ self.h + W["b_hn"]))
        self.h = (1 - z) * n + z * self.h
        self.n += w

    def p(self, prior: float) -> tuple[float, float]:
        pr = min(1 - 1e-6, max(1e-6, prior))
        logit = float(self.w["w_out"] @ self.h + self.w["b_out"][0]) + math.log(pr / (1 - pr))
        return 1 / (1 + math.exp(-logit)), self.n / (self.n + PRIOR_STRENGTH)


def make_belief(model: str, p: dict, clock=None) -> Belief:
    if model == "M1t":
        return TimeDecay(p["tau_s"], clock)
    if model == "M0":
        return Cumulative()
    if model == "M1":
        return Cumulative(gamma=p["gamma"])
    if model == "M2":
        return Window(int(round(p["window"])))
    if model == "M3":
        return Cumulative()  # strength set by the memory (empirical Bayes)
    if model == "M4":
        return ChangePoint(p["hazard"])
    if model == "M5":
        return HiddenState(p["q_low"], p["q_high"], p["switch"])
    if model == "M6":
        return Neural(p["weights"])
    raise ValueError(model)


@dataclass
class BeliefMemory:
    model: str = "M0"
    params: dict = field(default_factory=dict)
    evidence: str = "all"     # "all" | "exposed"

    def __post_init__(self) -> None:
        if self.evidence not in ("all", "exposed"):
            raise ValueError(self.evidence)
        self.pool = Cumulative()
        self.beliefs: dict[str, Belief] = {}
        self.open: dict[str, Encounter] = {}
        self.last_seen: dict[str, float] = {}
        self.log: list[tuple] = []  # (end time, identity, confidence, class, reliable, p_before) — evaluation only
        self.counts = dict.fromkeys((*CLASSES, "unreliable"), 0)
        self.now = 0.0

    # --------------------------------------------------------------- stream
    def observe(self, obs, t: float) -> None:
        self.now = t
        careful_action = obs.action in (SLOW, YIELD, DETOUR)
        for e in obs.entities:
            if e.kind != "person":
                continue
            enc = self.open.get(e.track)
            if enc is None:
                enc = self.open[e.track] = Encounter(e.track, start=t)
                enc.p_start = self.p_collision(e.identity, "person")[0] if e.identity else None
            if e.identity is not None and e.identity_confidence >= enc.confidence:
                enc.identity, enc.confidence = e.identity, e.identity_confidence
            if e.distance < EXPOSED_M and abs(obs.v) > MOVING_MPS:
                enc.exposed = True
            if careful_action and e.distance < AVOID_M:
                enc.avoided = True
            self.last_seen[e.track] = t
        for ev in obs.events:
            if ev[0] == "collision" and ev[1] in self.open:
                self.open[ev[1]].collision = True
        self.expire(t)

    def expire(self, t: float) -> None:
        for track in [k for k, s in self.last_seen.items() if t - s > ENTITY_TTL_S]:
            del self.last_seen[track]
            enc = self.open.pop(track, None)
            if enc is not None:
                self._end(enc, t)

    def _end(self, enc: Encounter, t: float) -> None:
        klass = enc.klass()
        self.counts[klass] += 1
        if not enc.reliable:
            self.counts["unreliable"] += 1
        self.log.append((round(t, 2), enc.identity, round(enc.confidence, 3), klass, enc.reliable,
                         getattr(enc, "p_start", None)))
        if self.evidence == "exposed" and klass not in ("collision", "exposed"):
            return
        y = 1.0 if enc.collision else 0.0
        prior = self._pool_mean()
        self.pool.update(y, 1.0, BASE_RATE)
        if enc.identity is None or enc.confidence <= 0 or (self.evidence == "exposed" and not enc.reliable):
            return
        b = self.beliefs.get(enc.identity)
        if b is None:
            b = self.beliefs[enc.identity] = make_belief(self.model, self.params, clock=lambda: self.now)
        b.update(y, enc.confidence, prior)
        if self.model == "M3":
            self._empirical_bayes()

    # --------------------------------------------------------------- beliefs
    def _pool_mean(self) -> float:
        return self.pool.p(BASE_RATE)[0]

    def _empirical_bayes(self) -> None:
        """M3: prior strength s from the dispersion of identity rates (method of moments, clipped to [0.5, 50])."""
        rates = [(b.k / b.n, b.n) for b in self.beliefs.values() if isinstance(b, Cumulative) and b.n >= 1]
        if len(rates) < 3:
            return
        m = float(np.average([r for r, _ in rates], weights=[n for _, n in rates]))
        var = float(np.average([(r - m) ** 2 for r, _ in rates], weights=[n for _, n in rates]))
        s = m * (1 - m) / var - 1 if var > 1e-9 else 50.0
        s = min(50.0, max(0.5, s))
        for b in self.beliefs.values():
            b.strength = s

    def p_collision(self, identity: str | None, kind: str) -> tuple[float, float]:
        prior = self._pool_mean()
        if kind != "person" or identity is None or identity not in self.beliefs:
            n = self.pool.n
            return prior, n / (n + PRIOR_STRENGTH)
        return self.beliefs[identity].p(prior)

    def state_size(self) -> int:
        """Floats held per identity (memory requirement)."""
        b = next(iter(self.beliefs.values()), None)
        if b is None:
            return 0
        if isinstance(b, Window):
            return 2 * b.trials.maxlen
        if isinstance(b, ChangePoint):
            return 3 * ChangePoint.RMAX
        if isinstance(b, HiddenState):
            return 2
        if isinstance(b, Neural):
            return int(b.h.size) + 1
        return 2


def entropy_of(counts: dict) -> float:
    total = sum(counts.values())
    return -sum(c / total * math.log(c / total) for c in counts.values() if c) if total else 0.0
