"""Entity and context outcome statistics, learned only from observed outcomes.

Teaching signal = OutcomeSignal (what happened), weighted by evidence quality and identity confidence. Never the
APM's own affect, and never an OutcomeHypothesis: a predicted collision that did not happen teaches nothing about
collisions (no counterfactual learning, no self-reinforcing fear loop).

Unit of evidence = an *encounter*: the lifetime of a track. When it ends, every outcome kind the embodiment can
detect gets one Bernoulli trial (happened or not), credited to
  - the bound persistent identity with weight = identity confidence (ambiguous identities get partial credit), and
  - the entity-type pool with weight 1 (generalisation to unfamiliar individuals).
Expectations are Beta-posterior means with the type pool as prior for individuals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from apm.core.outcomes import ADVERSE, BENEFICIAL

PRIOR_STRENGTH = 2.0
BASE_RATE = 0.05
FAMILIARITY_TAU_S = 60.0
MAX_RECORDS = 1024


@dataclass
class Record:
    encounters: float = 0.0
    counts: dict[str, float] = field(default_factory=dict)
    exposure_s: float = 0.0
    last_seen_s: float = 0.0


@dataclass
class Encounter:
    identity: str | None
    identity_confidence: float
    entity_type: str
    start_s: float
    outcomes: dict[str, float] = field(default_factory=dict)  # kind → credited evidence (0..1)


class EntityMemory:
    def __init__(self, detectable: tuple[str, ...]):
        self.detectable = tuple(detectable)
        self.records: dict[str, Record] = {}
        self.encounters: dict[str, Encounter] = {}  # active, by track id
        self.updates = 0

    # -------------------------------------------------- encounters
    def track(self, track_id: str, identity: str | None, identity_confidence: float, entity_type: str,
              now_s: float) -> None:
        e = self.encounters.get(track_id)
        if e is None:
            self.encounters[track_id] = Encounter(identity, identity_confidence, entity_type, now_s)
        elif identity is not None and identity_confidence >= e.identity_confidence:
            e.identity, e.identity_confidence = identity, identity_confidence

    def outcome(self, track_id: str | None, kind: str, evidence: float) -> bool:
        """Attribute an observed outcome to an active encounter. Returns False if it cannot be attributed."""
        if track_id is None or track_id not in self.encounters or kind not in self.detectable:
            return False
        e = self.encounters[track_id]
        e.outcomes[kind] = max(e.outcomes.get(kind, 0.0), evidence)
        return True

    def end(self, track_id: str, now_s: float) -> None:
        e = self.encounters.pop(track_id, None)
        if e is None:
            return
        duration = max(0.0, now_s - e.start_s)
        targets = [(f"type:{e.entity_type}", 1.0)]
        if e.identity is not None and e.identity_confidence > 0:
            targets.append((f"id:{e.identity}", e.identity_confidence))
        for key, weight in targets:
            record = self._record(key)
            record.encounters += weight
            record.exposure_s += weight * duration
            record.last_seen_s = now_s
            for kind in self.detectable:
                record.counts[kind] = record.counts.get(kind, 0.0) + weight * e.outcomes.get(kind, 0.0)
        self.updates += 1

    def _record(self, key: str) -> Record:
        if key not in self.records:
            if len(self.records) >= MAX_RECORDS:
                oldest = min((k for k in self.records if k.startswith("id:")), default=None,
                             key=lambda k: self.records[k].last_seen_s)
                if oldest is not None:
                    del self.records[oldest]
            self.records[key] = Record()
        return self.records[key]

    # -------------------------------------------------- expectations
    def p(self, kind: str, identity: str | None, entity_type: str) -> tuple[float, float]:
        """(probability per encounter, confidence 0..1) for an outcome kind with this entity."""
        pool = self.records.get(f"type:{entity_type}")
        prior = BASE_RATE
        if pool is not None and pool.encounters > 0:
            prior = (pool.counts.get(kind, 0.0) + PRIOR_STRENGTH * BASE_RATE) / (pool.encounters + PRIOR_STRENGTH)
        record = self.records.get(f"id:{identity}") if identity else None
        source = record if record is not None else pool
        if source is None or source is pool:
            n = pool.encounters if pool else 0.0
            return prior, n / (n + PRIOR_STRENGTH)
        p = (source.counts.get(kind, 0.0) + PRIOR_STRENGTH * prior) / (source.encounters + PRIOR_STRENGTH)
        return p, source.encounters / (source.encounters + PRIOR_STRENGTH)

    def expectation(self, identity: str | None, entity_type: str) -> dict[str, float]:
        adverse = [self.p(k, identity, entity_type) for k in self.detectable if k in ADVERSE]
        benign = [self.p(k, identity, entity_type) for k in self.detectable if k in BENEFICIAL]
        p_adverse = 1.0 - math.prod(1.0 - p for p, _ in adverse) if adverse else 0.0
        p_benign = 1.0 - math.prod(1.0 - p for p, _ in benign) if benign else 0.0
        confidence = max([c for _, c in adverse + benign], default=0.0)
        record = self.records.get(f"id:{identity}") if identity else None
        exposure = record.exposure_s if record else 0.0
        return {"p_adverse": p_adverse, "p_benign": p_benign,
                "familiarity": 1.0 - math.exp(-exposure / FAMILIARITY_TAU_S),
                "predictability": 1.0 - 4.0 * p_adverse * (1.0 - p_adverse), "confidence": confidence}

    # -------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"detectable": list(self.detectable), "updates": self.updates,
                "records": {k: {"encounters": r.encounters, "counts": r.counts, "exposure_s": r.exposure_s,
                                "last_seen_s": r.last_seen_s} for k, r in self.records.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> EntityMemory:
        memory = cls(tuple(data["detectable"]))
        memory.updates = int(data["updates"])
        memory.records = {k: Record(v["encounters"], dict(v["counts"]), v["exposure_s"], v["last_seen_s"])
                          for k, v in data["records"].items()}
        return memory


class ContextMemory:
    """Outcome rates per context key (e.g. "goal_type|mode"), counted in fixed windows of ``window_s``."""

    def __init__(self, detectable: tuple[str, ...], window_s: float = 10.0):
        self.detectable, self.window_s = tuple(detectable), window_s
        self.windows: dict[str, float] = {}
        self.counts: dict[str, dict[str, float]] = {}
        self._current: tuple[str, float, dict[str, float]] | None = None
        self.updates = 0

    def tick(self, key: str, now_s: float) -> None:
        if self._current is not None and (self._current[0] != key or now_s - self._current[1] >= self.window_s):
            self._close()
        if self._current is None:
            self._current = (key, now_s, {})

    def outcome(self, kind: str, evidence: float) -> None:
        if self._current is not None and kind in self.detectable:
            seen = self._current[2]
            seen[kind] = max(seen.get(kind, 0.0), evidence)

    def _close(self) -> None:
        assert self._current is not None
        key, _, seen = self._current
        self.windows[key] = self.windows.get(key, 0.0) + 1.0
        counts = self.counts.setdefault(key, {})
        for kind in self.detectable:
            counts[kind] = counts.get(kind, 0.0) + seen.get(kind, 0.0)
        self._current = None
        self.updates += 1

    def p(self, kind: str, key: str) -> float:
        n = self.windows.get(key, 0.0)
        return (self.counts.get(key, {}).get(kind, 0.0) + PRIOR_STRENGTH * BASE_RATE) / (n + PRIOR_STRENGTH)

    def to_dict(self) -> dict:
        return {"detectable": list(self.detectable), "window_s": self.window_s, "windows": self.windows,
                "counts": self.counts, "updates": self.updates}

    @classmethod
    def from_dict(cls, data: dict) -> ContextMemory:
        memory = cls(tuple(data["detectable"]), data["window_s"])
        memory.windows, memory.counts, memory.updates = dict(data["windows"]), dict(data["counts"]), data["updates"]
        return memory
