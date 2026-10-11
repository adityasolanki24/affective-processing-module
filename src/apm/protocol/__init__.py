"""AffectLink v1: versioned, transport-independent messages between robots and an APM."""
from apm.protocol.codec import MAX_MESSAGE_BYTES, decode, encode, from_dict, to_dict
from apm.protocol.messages import *  # noqa: F403
from apm.protocol.messages import MESSAGE_TYPES, SCHEMA_VERSION
from apm.protocol.validation import ProtocolError, StreamGate, validate

__all__ = ["MAX_MESSAGE_BYTES", "MESSAGE_TYPES", "SCHEMA_VERSION", "ProtocolError", "StreamGate", "decode", "encode",
           "from_dict", "to_dict", "validate"]
