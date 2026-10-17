"""Revising memory: per-identity type beliefs from robot-observable encounter evidence.

A drop-in replacement for F's `ObsMemory` and G's `BeliefMemory`: same `observe`, `expire` and `p_collision`
interface, plus G's `log`, `counts` and `state_size`, so G's decision rule and instrumentation run unchanged.

- Evidence: `evidence.Encounter`; an encounter updates its identity's belief only if it is informative (not
  censored) and the identity is reliable (confidence ≥ 0.5).
- Belief: `belief.TypeBelief` under the fitted `TypeModel`; unknown or unreliable people get the population prior.
- `thompson=True`: during each open encounter, `p_collision` answers with the collision rate of one type sampled from
  the identity's posterior, drawn once per encounter. `posterior_p` always gives the posterior mean (used for
  evaluation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from apm_revise.belief import TypeBelief, TypeModel, readout
from apm_revise.evidence import ENTITY_TTL_S, Encounter

CLASSES = ("collision", "exposed", "avoided", "unexposed")


@dataclass
class RevisionMemory:
    model: TypeModel
    lam: float = 0.0
    thompson: bool = False
    seed: int = 0
    beliefs: dict[str, TypeBelief] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.open: dict[str, Encounter] = {}
        self.last_seen: dict[str, float] = {}
        # (end time, identity, confidence, G class, reliable, p at start, outcome category or None,
        #  log predictive probability of that category before the update or None) — evaluation and tuning only
        self.log: list[tuple] = []
        self.counts = dict.fromkeys((*CLASSES, "unreliable", "censored"), 0)
        self.now = 0.0
        self.rng = np.random.default_rng(self.seed)
        self.sampled: dict[str, tuple[str, int]] = {}  # identity → (track, sampled type)

    # --------------------------------------------------------------- stream
    def observe(self, obs, t: float) -> None:
        self.now = t
        for e in obs.entities:
            if e.kind != "person":
                continue
            enc = self.open.get(e.track)
            if enc is None:
                enc = self.open[e.track] = Encounter(e.track, start=t)
                enc.p_start = self.posterior_p(e.identity)[0] if e.identity else None
            if e.identity is not None and e.identity_confidence >= enc.confidence:
                enc.identity, enc.confidence = e.identity, e.identity_confidence
            enc.observe(e.distance, obs.v, obs.action)
            self.last_seen[e.track] = t
        for ev in obs.events:
            if ev[0] == "collision" and ev[1] in self.open:
                self.open[ev[1]].collision = True
        self.expire(t)

    def expire(self, t: float) -> None:
        self.now = max(self.now, t)
        for track in [k for k, s in self.last_seen.items() if t - s > ENTITY_TTL_S]:
            del self.last_seen[track]
            enc = self.open.pop(track, None)
            if enc is not None:
                self._end(enc, t)

    def _end(self, enc: Encounter, t: float) -> None:
        klass = enc.g_class()
        cat = enc.category(self.model.coding)
        self.counts[klass] += 1
        if not enc.reliable:
            self.counts["unreliable"] += 1
        if cat is None:
            self.counts["censored"] += 1
        logp = None
        if cat is not None and enc.reliable:
            logp = round(math.log(max(1e-12, float(self.posterior(enc.identity) @ self.model.E[:, cat]))), 5)
        self.log.append((round(t, 2), enc.identity, round(enc.confidence, 3), klass, enc.reliable, enc.p_start, cat,
                         logp))
        if cat is None or not enc.reliable:
            return
        assert enc.identity is not None
        b = self.beliefs.get(enc.identity)
        if b is None:
            b = self.beliefs[enc.identity] = TypeBelief(self.model, self.lam, t)
        b.update(cat, enc.confidence, t)

    # --------------------------------------------------------------- read-outs
    def posterior(self, identity: str | None) -> np.ndarray:
        b = self.beliefs.get(identity) if identity is not None else None
        return b.at(self.now) if b is not None else self.model.pi.copy()

    def posterior_p(self, identity: str | None) -> tuple[float, float]:
        return readout(self.model, self.posterior(identity))

    def p_collision(self, identity: str | None, kind: str) -> tuple[float, float]:
        if kind != "person":
            return readout(self.model, self.model.pi)
        if not self.thompson or identity is None:
            return self.posterior_p(identity)
        track = next((k for k, e in self.open.items() if e.identity == identity), None)
        if track is None:
            return self.posterior_p(identity)
        cached = self.sampled.get(identity)
        if cached is None or cached[0] != track:
            b = self.posterior(identity)
            cached = self.sampled[identity] = (track, int(self.rng.choice(len(b), p=b)))
        return float(self.model.r[cached[1]]), self.posterior_p(identity)[1]

    def state_size(self) -> int:
        """Floats held per identity: the type posterior and its time stamp."""
        return self.model.K + 1
