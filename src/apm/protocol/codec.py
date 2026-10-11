"""Canonical encoding for the local transport: AffectLink messages ↔ JSON-compatible dicts.

The dict layout follows the proto3 JSON mapping of proto/affectlink/v1 (field names unchanged, enums as names,
absent optionals omitted), with a top-level ``"type"`` naming the message. Binary protobuf encoding for
constrained links is documented in docs/affectlink.md; this module needs no protobuf runtime.
"""
from __future__ import annotations

import dataclasses
import json
import types
import typing
from enum import Enum
from typing import Any

from apm.protocol.messages import MESSAGE_TYPES, Header, Quantity, Status

MAX_MESSAGE_BYTES = 16_384  # local/Ethernet profile; see docs/affectlink.md for the CAN FD/UART compact profile


def to_dict(message: Any) -> dict[str, Any]:
    out = {"type": type(message).__name__}
    out.update(_encode(message))
    return out


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {}
        for f in dataclasses.fields(value):
            item = getattr(value, f.name)
            if item is None:
                continue  # proto3 optional: absent
            result[f.name] = _encode(item)
        return result
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


def encode(message: Any) -> bytes:
    data = json.dumps(to_dict(message), separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    return data


def decode(data: bytes) -> Any:
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    return from_dict(json.loads(data.decode("utf-8")))


def from_dict(data: dict[str, Any]) -> Any:
    data = dict(data)
    name = data.pop("type", None)
    if name not in MESSAGE_TYPES:
        raise ValueError(f"unknown AffectLink message type: {name!r}")
    return _decode(MESSAGE_TYPES[name], data)


def _decode(cls: Any, data: Any) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs = {}
    known = {f.name for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in known:
            continue  # forward compatibility: ignore unknown fields from newer minor versions
        kwargs[key] = _convert(hints[key], value)
    return cls(**kwargs)


def _convert(hint: Any, value: Any) -> Any:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        return None if value is None else _convert(args[0], value)
    if origin is tuple:
        inner = typing.get_args(hint)
        if len(inner) == 2 and inner[1] is Ellipsis:
            return tuple(_convert(inner[0], v) for v in value)
        return tuple(_convert(t, v) for t, v in zip(inner, value, strict=True))
    if origin is dict:
        return dict(value)
    if hint is Status:
        return Status(value)
    if dataclasses.is_dataclass(hint):
        return _decode(hint, value)
    if hint is float:
        return float(value)
    return value


def header_of(message: Any) -> Header:
    return message.header


__all__ = ["MAX_MESSAGE_BYTES", "Quantity", "decode", "encode", "from_dict", "header_of", "to_dict"]
