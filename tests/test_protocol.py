"""AffectLink v1: encoding, validation, missing-data semantics, ordering, resets, size limits."""
import dataclasses
import json

import pytest

from apm.adapters.base import Emitter
from apm.protocol import MAX_MESSAGE_BYTES, ProtocolError, StreamGate, decode, encode, validate
from apm.protocol.messages import (
    Header,
    OutcomeHypothesis,
    OutcomeSignal,
    Quantity,
    Reset,
    SelfState,
    SemanticCue,
    Status,
)


def test_every_input_roundtrips_exactly(robot):
    msgs = [robot.profile, robot.self_state(1.0), robot.goal(1.0), robot.entity(1.0, identity="ann", confidence=.8),
            robot.hypothesis(1.0), robot.outcome(1.0),
            SemanticCue(header=robot.emit.header("cues", 1.0), speaker="ann", meaning="greeting", confidence=.6),
            Reset(header=Header(source="testbot", stream="control", epoch=1, seq=0, stamp_s=1.0), reason="reboot")]
    for m in msgs:
        assert decode(encode(m)) == m


def test_missing_measurements_are_never_zero(robot):
    s = robot.self_state(1.0, stability=None)
    data = json.loads(encode(s))
    assert "value" not in data["stability_margin"] and data["stability_margin"]["status"] == "MISSING"
    with pytest.raises(ProtocolError):  # a value under a non-usable status is a silent substitution
        validate(dataclasses.replace(s, stability_margin=Quantity(value=0.0, unit="fraction", status=Status.MISSING)))
    with pytest.raises(ProtocolError):  # usable status without a value
        validate(dataclasses.replace(s, stability_margin=Quantity(value=None, unit="fraction", status=Status.VALID)))
    with pytest.raises(ProtocolError):  # fraction outside [0, 1]
        validate(dataclasses.replace(s, stability_margin=Quantity.measured(1.3, "fraction")))


def test_prediction_and_outcome_are_distinct_types(robot):
    h, o = robot.hypothesis(1.0), robot.outcome(1.0)
    assert type(h) is OutcomeHypothesis and type(o) is OutcomeSignal
    assert json.loads(encode(h))["type"] == "OutcomeHypothesis" and json.loads(encode(o))["type"] == "OutcomeSignal"


def test_schema_version_size_and_unknown_fields(robot):
    m = robot.goal(1.0)
    with pytest.raises(ProtocolError):
        validate(dataclasses.replace(m, header=dataclasses.replace(m.header, schema_version="affectlink/2.0")))
    data = json.loads(encode(m))
    data["future_field"] = 1  # newer minor version: ignored
    assert decode(json.dumps(data).encode()) == m
    with pytest.raises(ValueError):
        encode(SemanticCue(header=robot.emit.header("cues", 1.0), speaker="x", meaning="y" * MAX_MESSAGE_BYTES,
                           confidence=.5))


def test_stream_gate_order_duplicates_staleness_and_reset():
    e, gate = Emitter("bot"), StreamGate(max_age_s=1.0)
    first = SelfState(header=e.header("s", 10.0))
    assert gate.admit(first, 10.0) is None
    assert gate.admit(first, 10.0) == "duplicate_or_out_of_order"
    late = SelfState(header=e.header("s", 8.0))
    assert gate.admit(late, 10.0) == "stale"
    assert gate.admit(SelfState(header=e.header("s", 12.0)), 10.0) == "future_stamp"
    e.restart()
    assert gate.admit(Reset(header=e.header("control", 11.0)), 11.0) is None
    assert gate.admit(SelfState(header=e.header("s", 11.0)), 11.0) is None  # seq restarted in the new epoch
    old = SelfState(header=Header(source="bot", stream="s", epoch=0, seq=99, stamp_s=11.0))
    assert gate.admit(old, 11.0) == "old_epoch"
    assert gate.rejected == 4
