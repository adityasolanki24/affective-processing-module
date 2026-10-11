"""APM runtime: AffectLink messages in, advisory AffectLink messages out.

    ingest(message)  validates and orders the message, then routes it:
        perception      EntityObservation, RelationObservation, SemanticCue → bounded context (no affect change)
        self/goals      SelfState, GoalState                                → bounded context
        prediction      OutcomeHypothesis   → context (appraised every step as *tonic* context; never memory)
        outcome         OutcomeSignal       → appraisal → one phasic impulse + memory + episode (exactly once)
    step(now)        expires stale context, integrates affect to `now`, and returns the output bundle.

The runtime never commands actuators and is never a safety authority: a robot must behave safely with the APM
absent, late, or wrong (docs/safety.md).
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from apm.core.affect import MODEL_VERSION, AffectCore, tonic_input
from apm.core.appraisal import Frame, appraise_hypothesis, appraise_outcome
from apm.core.context import Context
from apm.core.outcomes import CATALOG
from apm.core.tendencies import TendencyFilter, raw_tendencies
from apm.memory.entities import ContextMemory, EntityMemory
from apm.memory.episodes import Episode, EpisodicMemory
from apm.protocol import codec
from apm.protocol.messages import (
    SCHEMA_VERSION,
    ActionTendency,
    AffectState,
    AppraisalFrame,
    AttentionBias,
    Diagnostics,
    EmbodimentProfile,
    EntityObservation,
    ExpressionIntent,
    GoalState,
    Header,
    MemoryExpectation,
    OutcomeHypothesis,
    OutcomeSignal,
    RelationObservation,
    Reset,
    SelfState,
    SemanticCue,
)
from apm.protocol.validation import StreamGate

STATE_VERSION = 1
OUTCOME_FRAME_TTL_S = 3.0
NOVELTY_TAU_S = 5.0
MAX_SEEN_OUTCOMES = 4096
MAX_CUES = 64
MEMORY_MIN_CONFIDENCE = 0.3
MEMORY_MIN_P = 0.1


def _triple(x) -> tuple[float, float, float]:
    return float(x[0]), float(x[1]), float(x[2])


@dataclass
class Outputs:
    affect: AffectState
    appraisals: list[AppraisalFrame]
    tendencies: list[ActionTendency]
    attention: list[AttentionBias]
    expression: ExpressionIntent
    expectations: list[MemoryExpectation]
    diagnostics: Diagnostics

    def affect_vector(self) -> np.ndarray:
        return np.array([self.affect.valence, self.affect.arousal, self.affect.dominance])

    def messages(self) -> list[Any]:
        return [self.affect, *self.appraisals, *self.tendencies, *self.attention, self.expression,
                *self.expectations, self.diagnostics]


@dataclass
class APM:
    profile: EmbodimentProfile
    source_id: str = "apm"
    max_age_s: float = 2.0
    context: Context = field(init=False)
    gate: StreamGate = field(init=False)
    affect: AffectCore = field(default_factory=AffectCore)
    entities: EntityMemory = field(init=False)
    contexts: ContextMemory = field(init=False)
    episodes: EpisodicMemory = field(default_factory=EpisodicMemory)
    tendency_filter: TendencyFilter = field(default_factory=TendencyFilter)
    outcome_frames: deque = field(default_factory=lambda: deque(maxlen=256))
    seen_outcomes: deque = field(default_factory=lambda: deque(maxlen=MAX_SEEN_OUTCOMES))
    cues: deque = field(default_factory=lambda: deque(maxlen=MAX_CUES))
    relations: deque = field(default_factory=lambda: deque(maxlen=MAX_CUES))
    seq: dict[str, int] = field(default_factory=dict)
    last_input_s: dict[str, float] = field(default_factory=dict)
    last_self_state_s: float | None = None

    def __post_init__(self) -> None:
        self.context = Context(profile=self.profile)
        self.gate = StreamGate(max_age_s=self.max_age_s)
        self.entities = EntityMemory(tuple(self.profile.outcome_kinds))
        self.contexts = ContextMemory(tuple(self.profile.outcome_kinds))
        unknown = [k for k in self.profile.outcome_kinds if k not in CATALOG]
        if unknown:
            raise ValueError(f"profile declares outcome kinds outside the catalog: {unknown}")

    # ------------------------------------------------------------------ input
    def ingest(self, message: Any, now_s: float | None = None) -> str | None:
        """Returns None when accepted, otherwise the rejection reason (also counted in diagnostics)."""
        now = message.header.stamp_s if now_s is None else now_s
        reason = self.gate.admit(message, now)
        if reason is not None:
            return reason
        h = message.header
        self.last_input_s[h.stream] = h.stamp_s
        if isinstance(message, SelfState):
            self.last_self_state_s = h.stamp_s
        if isinstance(message, Reset):
            self.context = Context(profile=self.profile)  # a restarted sender's context is no longer valid
        elif isinstance(message, EmbodimentProfile):
            return self._reject("profile_change_requires_new_apm") if message != self.profile else None
        elif isinstance(message, EntityObservation):
            self.context.observe_entity(message)
            self.entities.track(message.track_id, message.identity, message.identity_confidence,
                                message.entity_type, h.stamp_s)
        elif isinstance(message, SelfState):
            self.context.observe_self(message)
        elif isinstance(message, GoalState):
            self.context.observe_goal(message)
        elif isinstance(message, OutcomeHypothesis):
            if message.kind not in CATALOG:
                return self._reject("unknown_kind")
            self.context.observe_hypothesis(message)
        elif isinstance(message, OutcomeSignal):
            return self._outcome(message)
        elif isinstance(message, SemanticCue):
            self.cues.append(message)  # recorded; the baseline does not let interpreted cues drive affect
        elif isinstance(message, RelationObservation):
            self.relations.append(message)
        else:
            return self._reject("not_an_input")
        return None

    def _reject(self, reason: str) -> str:
        self.gate._reject(reason)
        return reason

    def _outcome(self, o: OutcomeSignal) -> str | None:
        if o.outcome_id in self.seen_outcomes:
            return self._reject("duplicate_outcome")
        if o.kind not in self.profile.outcome_kinds:
            return self._reject("undeclared_outcome_kind")
        self.seen_outcomes.append(o.outcome_id)
        now = o.header.stamp_s
        self.affect.advance(now)
        margins = self.context.body_margins(now)
        frame = appraise_outcome(o, margins, self._goal_priority(o.affected), self._predicted(o.kind, o.target, now),
                                 now)
        self.affect.impulse(frame)
        self.outcome_frames.append(frame)
        evidence = o.evidence_quality * (1.0 if o.confirmed else 0.7)
        attributed = self.entities.outcome(o.target, o.kind, evidence)
        self.contexts.outcome(o.kind, evidence)
        track = self.context.tracks.get(o.target) if o.target else None
        identity = track.observation.identity if track else None
        self.episodes.add(Episode(
            stamp_s=now, outcome_id=o.outcome_id, kind=o.kind, magnitude=o.magnitude, confirmed=o.confirmed,
            evidence_quality=o.evidence_quality, target=o.target, identity=identity,
            identity_confidence=track.observation.identity_confidence if track else 0.0,
            predicted_probability=self._live_probability(o.kind, o.target), action_context=o.action_context,
            affect=_triple(self.affect.p), margins=margins))
        if o.target is not None and o.target != "self" and not attributed:
            return None  # accepted; counted in context memory only (target unknown or not tracked)
        return None

    def _goal_priority(self, affected: str) -> float | None:
        return self.context.active_goal_priority(affected)

    def _live_probability(self, kind: str, target: str | None) -> float | None:
        ps = [h.probability for h in self.context.hypotheses.values() if h.kind == kind
              and (target is None or h.target in (None, target))]
        return max(ps) if ps else None

    def _memory_probability(self, kind: str, target: str | None) -> float:
        track = self.context.tracks.get(target) if target else None
        if track is not None:
            return self.entities.p(kind, track.observation.identity, track.observation.entity_type)[0]
        return self.contexts.p(kind, self._context_key(self.affect.t or 0.0))

    def _predicted(self, kind: str, target: str | None, now_s: float) -> float:
        live = self._live_probability(kind, target)
        return live if live is not None else self._memory_probability(kind, target)

    def _context_key(self, now_s: float) -> str:
        goals = sorted((g for g in self.context.goals.values() if g.active), key=lambda g: -g.priority)
        mode = self.context.self_state.mode if self.context.self_state else ""
        return f"{goals[0].goal_type if goals else 'none'}|{mode}"

    # ------------------------------------------------------------------ step
    def step(self, now_s: float) -> Outputs:
        for track in self.context.expire(now_s):
            self.entities.end(track.observation.track_id, now_s)
        self.contexts.tick(self._context_key(now_s), now_s)
        margins = self.context.body_margins(now_s)
        missing = self.context.missing_capabilities(now_s)
        frames = [appraise_hypothesis(h, margins, self._goal_priority(h.affected),
                                      self._memory_probability(h.kind, h.target), now_s)
                  for h in self.context.hypotheses.values()]
        frames += self._memory_frames(margins, now_s)
        self.affect.advance(now_s)
        self.affect.set_context(tonic_input(frames, margins, missing_margin=not margins))
        while self.outcome_frames and now_s - self.outcome_frames[0].stamp_s > OUTCOME_FRAME_TTL_S:
            self.outcome_frames.popleft()
        recent = list(self.outcome_frames)
        novelty, p_adverse, expectations = {}, {}, []
        for track_id, track in self.context.tracks.items():
            obs = track.observation
            e = self.entities.expectation(obs.identity, obs.entity_type)
            novelty[track_id] = (1.0 - e["familiarity"]) * math.exp(-(now_s - track.first_seen_s) / NOVELTY_TAU_S)
            p_adverse[track_id] = e["p_adverse"]
            expectations.append(MemoryExpectation(header=self._header("expectation", now_s), target=track_id,
                                                  context=obs.identity or f"type:{obs.entity_type}", **e))
        strain = 1.0 - min(margins.values()) if margins else 0.0
        tendencies = self.tendency_filter.update(
            raw_tendencies(frames + recent, novelty, p_adverse, strain, margins.get("stability_margin")), now_s)
        return Outputs(
            affect=self._affect_message(now_s),
            appraisals=[self._frame_message(f, now_s) for f in frames + recent],
            tendencies=[ActionTendency(header=self._header("tendency", now_s), target=t, **v)
                        for t, v in sorted(tendencies.items())],
            attention=self._attention(tendencies, now_s),
            expression=self._expression(now_s, tendencies),
            expectations=expectations,
            diagnostics=self._diagnostics(now_s, missing))

    def _memory_frames(self, margins: dict[str, float], now_s: float) -> list[Frame]:
        """Memory's own predictions for tracked entities: P(kind | this identity/type) per encounter.

        Internal hypotheses (reference_type "memory"), appraised as tonic context like any prediction and, like any
        prediction, never written back to memory. Only kinds with evidence above the base rate are raised.
        """
        frames = []
        for track_id, track in self.context.tracks.items():
            obs = track.observation
            for kind in self.profile.outcome_kinds:
                p, confidence = self.entities.p(kind, obs.identity, obs.entity_type)
                if confidence < MEMORY_MIN_CONFIDENCE or p <= MEMORY_MIN_P:
                    continue
                h = OutcomeHypothesis(header=self._header("memory", now_s), hypothesis_id=f"memory:{track_id}:{kind}",
                                      kind=kind, target=track_id, affected="self", probability=p,
                                      horizon_s=self.context.entity_ttl_s, confidence=confidence * obs.confidence)
                f = appraise_hypothesis(h, margins, None, p, now_s)
                frames.append(Frame(**{**f.__dict__, "reference_type": "memory"}))
        return frames

    # ------------------------------------------------------------------ output helpers
    def _header(self, stream: str, now_s: float) -> Header:
        self.seq[stream] = self.seq.get(stream, -1) + 1
        return Header(schema_version=SCHEMA_VERSION, source=self.source_id, stream=stream, seq=self.seq[stream],
                      stamp_s=now_s)

    def _affect_message(self, now_s: float) -> AffectState:
        p = self.affect.p
        unknown = float(self.affect.u[4])
        return AffectState(header=self._header("affect", now_s), valence=float(p[0]), arousal=float(p[1]),
                           dominance=float(p[2]), baseline=_triple(self.affect.baseline),
                           uncertainty=unknown, model_version=MODEL_VERSION)

    def _frame_message(self, f: Frame, now_s: float) -> AppraisalFrame:
        return AppraisalFrame(header=self._header("appraisal", now_s), reference_id=f.reference_id,
                              reference_type=f.reference_type, kind=f.kind, target=f.target, relevance=f.relevance,
                              desirability=f.desirability, likelihood=f.likelihood, expectedness=f.expectedness,
                              controllability=f.controllability, uncertainty=f.uncertainty)

    def _attention(self, tendencies: dict[str, dict[str, float]], now_s: float) -> list[AttentionBias]:
        scored = sorted(((max(v["avoid"], v["disengage"], v["approach"], v["orient"]), t)
                         for t, v in tendencies.items() if t != "self"), reverse=True)
        return [AttentionBias(header=self._header("attention", now_s), target=t, priority=float(s),
                              expires_s=now_s + 1.0) for s, t in scored[:3] if s > 0.05]

    def _expression(self, now_s: float, tendencies: dict[str, dict[str, float]]) -> ExpressionIntent:
        v, a, d = (float(x) for x in self.affect.p)
        urgency = max([max(t["avoid"], t["disengage"]) for t in tendencies.values()], default=0.0)
        return ExpressionIntent(header=self._header("expression", now_s), activation=(a + 1) / 2,
                                posture=float(np.clip(0.5 * v + 0.5 * d, -1, 1)),
                                movement_energy=float(np.clip((a + 1) / 2 * (0.75 + 0.25 * d), 0, 1)),
                                openness=v, urgency=float(urgency))

    def _diagnostics(self, now_s: float, missing: list[str]) -> Diagnostics:
        ages = {k: now_s - t for k, t in self.last_input_s.items()}
        # Event streams (outcomes, hypotheses, entities) are legitimately silent; only the periodic self state
        # has a freshness requirement.
        stale = self.last_self_state_s is None or now_s - self.last_self_state_s > self.max_age_s
        health = "stale_inputs" if stale else ("degraded" if missing else "ok")
        return Diagnostics(header=self._header("diagnostics", now_s), input_age_s=ages,
                           saturation=self.affect.saturation(), missing_capabilities=tuple(missing),
                           model_version=MODEL_VERSION,
                           memory_updates=self.entities.updates + self.contexts.updates,
                           health=health, rejected_messages=self.gate.rejected)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        state = {"state_version": STATE_VERSION, "model_version": MODEL_VERSION,
                 "profile": codec.to_dict(self.profile), "affect": self.affect.to_dict(),
                 "entities": self.entities.to_dict(), "contexts": self.contexts.to_dict(),
                 "episodes": self.episodes.to_dict(), "seen_outcomes": list(self.seen_outcomes)}
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX and Windows: a crash mid-save leaves the previous state intact

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> APM:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("state_version") != STATE_VERSION:
            raise ValueError("incompatible APM state version")
        apm = cls(profile=codec.from_dict(state["profile"]), **kwargs)
        apm.affect = AffectCore.from_dict(state["affect"])
        apm.entities = EntityMemory.from_dict(state["entities"])
        apm.contexts = ContextMemory.from_dict(state["contexts"])
        apm.episodes = EpisodicMemory.from_dict(state["episodes"])
        apm.seen_outcomes.extend(state["seen_outcomes"])
        return apm


def load_or_new(path: str | Path, profile: EmbodimentProfile, **kwargs: Any) -> tuple[APM, str | None]:
    """Restore saved state if it is intact and belongs to this embodiment, else start empty.

    Returns (apm, reason): reason is None on a clean restore, otherwise why the state was not used. Partial or
    corrupt state is never used.
    """
    path = Path(path)
    if not path.exists():
        return APM(profile=profile, **kwargs), "no_saved_state"
    try:
        apm = APM.load(path, **kwargs)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return APM(profile=profile, **kwargs), f"corrupt_or_incompatible_state:{type(error).__name__}"
    if apm.profile.robot_id != profile.robot_id or apm.profile.outcome_kinds != profile.outcome_kinds:
        return APM(profile=profile, **kwargs), "state_from_another_embodiment"
    return apm, None
