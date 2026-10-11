"""AffectLink v1 messages: Python reference implementation of proto/affectlink/v1/affectlink.proto.

Field names and meanings match the .proto exactly (tests/test_proto_sync.py). Optional fields are ``None`` when
absent; a ``Quantity`` that is not available has ``value=None`` and a non-usable status, never zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = "affectlink/1.0"
SCHEMA_MAJOR = 1


class Status(str, Enum):
    STATUS_UNSPECIFIED = "STATUS_UNSPECIFIED"
    VALID = "VALID"
    ESTIMATED = "ESTIMATED"
    MISSING = "MISSING"
    STALE = "STALE"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"


USABLE = (Status.VALID, Status.ESTIMATED)


@dataclass(frozen=True, kw_only=True)
class Header:
    schema_version: str = SCHEMA_VERSION
    source: str
    stream: str
    epoch: int = 0
    seq: int
    stamp_s: float


@dataclass(frozen=True, kw_only=True)
class Quantity:
    value: float | None = None
    unit: str
    sigma: float | None = None
    status: Status = Status.MISSING

    @property
    def usable(self) -> bool:
        return self.status in USABLE and self.value is not None and math.isfinite(self.value)

    @staticmethod
    def missing(unit: str, status: Status = Status.MISSING) -> Quantity:
        return Quantity(value=None, unit=unit, status=status)

    @staticmethod
    def measured(value: float, unit: str, sigma: float | None = None, estimated: bool = False) -> Quantity:
        return Quantity(value=float(value), unit=unit, sigma=sigma,
                        status=Status.ESTIMATED if estimated else Status.VALID)


@dataclass(frozen=True, kw_only=True)
class Provenance:
    origin: str = ""
    method: str = "measured"
    model_version: str = ""


@dataclass(frozen=True, kw_only=True)
class Vector3:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, kw_only=True)
class MarginCalibration:
    field: str
    raw_unit: str
    nominal: float
    critical: float
    meaning: str = ""

    def to_margin(self, raw: float) -> float:
        """Raw value → margin fraction 0..1 (1 = nominal, 0 = critical), clipped."""
        span = self.nominal - self.critical
        if span == 0:
            raise ValueError("degenerate calibration")
        return min(1.0, max(0.0, (raw - self.critical) / span))


@dataclass(frozen=True, kw_only=True)
class EmbodimentProfile:
    header: Header
    robot_id: str
    embodiment_class: str
    self_fields: tuple[str, ...] = ()
    outcome_kinds: tuple[str, ...] = ()
    goal_types: tuple[str, ...] = ()
    calibration: tuple[MarginCalibration, ...] = ()


@dataclass(frozen=True, kw_only=True)
class EntityObservation:
    header: Header
    track_id: str
    identity: str | None = None
    identity_confidence: float = 0.0
    entity_type: str
    position: Vector3 | None = None
    bearing: Quantity = field(default_factory=lambda: Quantity.missing("rad"))
    distance: Quantity = field(default_factory=lambda: Quantity.missing("m"))
    closing_speed: Quantity = field(default_factory=lambda: Quantity.missing("m/s"))
    confidence: float = 1.0
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(frozen=True, kw_only=True)
class RelationObservation:
    header: Header
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    provenance: Provenance = field(default_factory=Provenance)


SELF_FIELDS = ("energy_margin", "thermal_margin", "stability_margin", "actuation_stress", "contact_slip",
               "sensor_health")


@dataclass(frozen=True, kw_only=True)
class SelfState:
    header: Header
    energy_margin: Quantity = field(default_factory=lambda: Quantity.missing("fraction"))
    thermal_margin: Quantity = field(default_factory=lambda: Quantity.missing("fraction"))
    stability_margin: Quantity = field(default_factory=lambda: Quantity.missing("fraction"))
    actuation_stress: Quantity = field(default_factory=lambda: Quantity.missing("fraction"))
    contact_slip: Quantity = field(default_factory=lambda: Quantity.missing("fraction"))
    sensor_health: Quantity = field(default_factory=lambda: Quantity.missing("fraction"))
    mode: str = ""
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(frozen=True, kw_only=True)
class GoalState:
    header: Header
    goal_id: str
    goal_type: str
    priority: float
    target: str | None = None
    progress: Quantity = field(default_factory=lambda: Quantity.missing("fraction"))
    active: bool = True


@dataclass(frozen=True, kw_only=True)
class OutcomeHypothesis:
    header: Header
    hypothesis_id: str
    kind: str
    target: str | None = None
    affected: str
    probability: float
    horizon_s: float
    expected_magnitude: float = 1.0
    confidence: float = 1.0
    action_context: str | None = None
    provenance: Provenance = field(default_factory=lambda: Provenance(method="predicted"))


@dataclass(frozen=True, kw_only=True)
class OutcomeSignal:
    header: Header
    outcome_id: str
    kind: str
    magnitude: float
    target: str | None = None
    affected: str
    evidence_quality: float = 1.0
    action_context: str | None = None
    confirmed: bool = True
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(frozen=True, kw_only=True)
class SemanticCue:
    header: Header
    speaker: str
    target: str | None = None
    meaning: str
    confidence: float
    provenance: Provenance = field(default_factory=lambda: Provenance(method="interpreted"))


@dataclass(frozen=True, kw_only=True)
class Reset:
    header: Header
    reason: str = ""


# ---------------------------------------------------------------- outputs

@dataclass(frozen=True, kw_only=True)
class AffectState:
    header: Header
    valence: float
    arousal: float
    dominance: float
    baseline: tuple[float, float, float]
    uncertainty: float
    model_version: str


@dataclass(frozen=True, kw_only=True)
class AppraisalFrame:
    header: Header
    reference_id: str
    reference_type: str
    kind: str
    target: str | None
    relevance: float
    desirability: float
    likelihood: float
    expectedness: float
    controllability: float
    uncertainty: float


@dataclass(frozen=True, kw_only=True)
class ActionTendency:
    header: Header
    target: str
    approach: float = 0.0
    avoid: float = 0.0
    orient: float = 0.0
    explore: float = 0.0
    disengage: float = 0.0
    seek_assistance: float = 0.0


@dataclass(frozen=True, kw_only=True)
class AttentionBias:
    header: Header
    target: str
    priority: float
    expires_s: float


@dataclass(frozen=True, kw_only=True)
class ExpressionIntent:
    header: Header
    activation: float
    posture: float
    movement_energy: float
    openness: float
    urgency: float


@dataclass(frozen=True, kw_only=True)
class MemoryExpectation:
    header: Header
    target: str
    context: str
    p_adverse: float
    p_benign: float
    familiarity: float
    predictability: float
    confidence: float


@dataclass(frozen=True, kw_only=True)
class Diagnostics:
    header: Header
    input_age_s: dict[str, float]
    saturation: dict[str, float]
    missing_capabilities: tuple[str, ...]
    model_version: str
    memory_updates: int
    health: str
    rejected_messages: int


INPUT_TYPES = (EmbodimentProfile, EntityObservation, RelationObservation, SelfState, GoalState, OutcomeHypothesis,
               OutcomeSignal, SemanticCue, Reset)
OUTPUT_TYPES = (AffectState, AppraisalFrame, ActionTendency, AttentionBias, ExpressionIntent, MemoryExpectation,
                Diagnostics)
MESSAGE_TYPES = {cls.__name__: cls for cls in INPUT_TYPES + OUTPUT_TYPES}
