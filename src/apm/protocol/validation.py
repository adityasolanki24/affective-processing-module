"""AffectLink message validation and stream ordering (docs/affectlink.md §Protocol rules).

``validate`` checks one message in isolation (schema major version, ranges, missing-data semantics).
``StreamGate`` enforces per-stream ordering, duplicate rejection, reset semantics and staleness on arrival.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any

from apm.protocol.messages import (
    SCHEMA_MAJOR,
    USABLE,
    EntityObservation,
    GoalState,
    OutcomeHypothesis,
    OutcomeSignal,
    Quantity,
    Reset,
    SemanticCue,
    Status,
)

UNIT_RANGES = {"fraction": (0.0, 1.0)}


class ProtocolError(ValueError):
    pass


def _unit_interval(name: str, value: float) -> None:
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ProtocolError(f"{name} must be within [0, 1], got {value!r}")


def _quantity(name: str, q: Quantity) -> None:
    if q.status is Status.STATUS_UNSPECIFIED:
        raise ProtocolError(f"{name}: status must be specified")
    if q.status in USABLE:
        if q.value is None or not math.isfinite(q.value):
            raise ProtocolError(f"{name}: usable status requires a finite value")
        low_high = UNIT_RANGES.get(q.unit)
        if low_high and not low_high[0] <= q.value <= low_high[1]:
            raise ProtocolError(f"{name}: {q.value} outside {q.unit} range {low_high}")
    elif q.value is not None:
        raise ProtocolError(f"{name}: status {q.status.value} must not carry a value (no silent substitution)")
    if q.sigma is not None and (not math.isfinite(q.sigma) or q.sigma < 0):
        raise ProtocolError(f"{name}: sigma must be finite and non-negative")
    if not q.unit:
        raise ProtocolError(f"{name}: unit is required")


def validate(message: Any) -> None:
    h = message.header
    try:
        major = int(h.schema_version.split("/")[1].split(".")[0])
    except (IndexError, ValueError) as error:
        raise ProtocolError(f"malformed schema_version {h.schema_version!r}") from error
    if not h.schema_version.startswith("affectlink/") or major != SCHEMA_MAJOR:
        raise ProtocolError(f"unsupported schema {h.schema_version!r}")
    if not h.source or not h.stream or h.seq < 0 or h.epoch < 0 or not math.isfinite(h.stamp_s):
        raise ProtocolError("header requires source, stream, non-negative seq/epoch and a finite stamp")
    for f in fields(message):
        value = getattr(message, f.name)
        if isinstance(value, Quantity):
            _quantity(f.name, value)
    if isinstance(message, EntityObservation):
        _unit_interval("confidence", message.confidence)
        _unit_interval("identity_confidence", message.identity_confidence)
        if message.identity is None and message.identity_confidence > 0:
            raise ProtocolError("identity_confidence without identity")
    elif isinstance(message, GoalState):
        _unit_interval("priority", message.priority)
    elif isinstance(message, OutcomeHypothesis):
        for name in ("probability", "expected_magnitude", "confidence"):
            _unit_interval(name, getattr(message, name))
        if not (math.isfinite(message.horizon_s) and message.horizon_s > 0):
            raise ProtocolError("horizon_s must be positive")
    elif isinstance(message, OutcomeSignal):
        for name in ("magnitude", "evidence_quality"):
            _unit_interval(name, getattr(message, name))
    elif isinstance(message, SemanticCue):
        _unit_interval("confidence", message.confidence)


@dataclass
class StreamGate:
    """Per-(source, stream) ordering: epoch first, then seq. Older or duplicate messages are rejected.

    A Reset from a source opens a new epoch for all its streams; anything from an older epoch is then rejected.
    """
    max_age_s: float = 2.0
    last: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)
    epochs: dict[str, int] = field(default_factory=dict)
    rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def admit(self, message: Any, now_s: float) -> str | None:
        """Returns None if accepted, else the rejection reason."""
        try:
            validate(message)
        except ProtocolError:
            return self._reject("invalid")
        h = message.header
        current_epoch = self.epochs.get(h.source, 0)
        if h.epoch < current_epoch:
            return self._reject("old_epoch")
        if isinstance(message, Reset):
            if h.epoch <= current_epoch and h.source in self.epochs:
                return self._reject("reset_without_new_epoch")
            self.epochs[h.source] = h.epoch
            for key in [k for k in self.last if k[0] == h.source]:
                del self.last[key]
            return None
        if h.epoch > current_epoch:
            self.epochs[h.source] = h.epoch
        key = (h.source, h.stream)
        previous = self.last.get(key)
        if previous is not None and (h.epoch, h.seq) <= previous:
            return self._reject("duplicate_or_out_of_order")
        if now_s - h.stamp_s > self.max_age_s:
            return self._reject("stale")
        if h.stamp_s - now_s > self.max_age_s:
            return self._reject("future_stamp")
        self.last[key] = (h.epoch, h.seq)
        return None

    def _reject(self, reason: str) -> str:
        self.rejected += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return reason
