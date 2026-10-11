"""The Python reference messages must match proto/affectlink/v1/affectlink.proto field by field."""
import dataclasses
import re
from pathlib import Path

from apm.protocol.messages import MESSAGE_TYPES, MarginCalibration, Provenance, Quantity, Vector3

PROTO = Path(__file__).resolve().parents[1] / "proto" / "affectlink" / "v1" / "affectlink.proto"


def proto_messages() -> dict[str, list[str]]:
    text = re.sub(r"//.*", "", PROTO.read_text(encoding="utf-8"))
    out = {}
    for name, body in re.findall(r"message\s+(\w+)\s*\{(.*?)\}", text, re.S):
        out[name] = re.findall(r"(?:optional\s+|repeated\s+)?(?:map<[^>]+>|[\w.]+)\s+(\w+)\s*=\s*\d+\s*;", body)
    return out


def test_every_message_and_field_matches_the_proto():
    proto = proto_messages()
    classes = {**MESSAGE_TYPES, "Quantity": Quantity, "Provenance": Provenance, "Vector3": Vector3,
               "MarginCalibration": MarginCalibration}
    assert set(proto) - {"Header"} == set(classes)
    for name, cls in classes.items():
        assert [f.name for f in dataclasses.fields(cls)] == proto[name], name


def test_field_numbers_are_unique_per_message():
    text = re.sub(r"//.*", "", PROTO.read_text(encoding="utf-8"))
    for name, body in re.findall(r"message\s+(\w+)\s*\{(.*?)\}", text, re.S):
        numbers = re.findall(r"=\s*(\d+)\s*;", body)
        assert len(numbers) == len(set(numbers)), name
