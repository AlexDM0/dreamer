"""Kid WebSocket wire protocol: binary frame header and JSON message helpers.

Binary frames are a 5-byte header (type: uint8, seq: uint32 big-endian) followed by
the payload. Text frames are JSON objects with a ``t`` field.
"""
from __future__ import annotations

import json
import struct
from typing import Any

SKETCH = 0x01   # client -> server, PNG payload
PICTURE = 0x02  # server -> client, JPEG payload

HEADER = struct.Struct(">BI")
HEADER_SIZE = HEADER.size  # 5
MAX_SEQ = 0xFFFFFFFF
MAX_MESSAGE_BYTES = 2 * 1024 * 1024
PROTOCOL_VERSION = 1

# Kid-socket fields that are dropped on sight: the kid page never supplies prompt text.
FORBIDDEN_FIELDS = frozenset({"prompt", "text"})


class ProtocolError(ValueError):
    """A malformed binary frame."""


def pack(msg_type: int, seq: int, payload: bytes) -> bytes:
    """Build a binary frame. ``seq`` must fit in uint32."""
    if not 0 <= seq <= MAX_SEQ:
        raise ValueError(f"seq out of range: {seq}")
    return HEADER.pack(msg_type, seq) + payload


def unpack(data: bytes) -> tuple[int, int, bytes]:
    """Split a binary frame into ``(type, seq, payload)``; raises ProtocolError."""
    if len(data) < HEADER_SIZE:
        raise ProtocolError(f"short frame ({len(data)} bytes)")
    msg_type, seq = HEADER.unpack_from(data)
    if msg_type not in (SKETCH, PICTURE):
        raise ProtocolError(f"unknown frame type 0x{msg_type:02x}")
    return msg_type, seq, bytes(data[HEADER_SIZE:])


def pack_picture(seq: int, jpeg: bytes) -> bytes:
    return pack(PICTURE, seq, jpeg)


def parse_client_text(text: str) -> dict[str, Any] | None:
    """Parse a kid-socket text message.

    Returns a dict with a string ``t`` and with any prompt/text fields removed, or None
    for anything that is not a JSON object with a string ``t``.
    """
    if len(text) > MAX_MESSAGE_BYTES:
        return None
    try:
        msg = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(msg, dict) or not isinstance(msg.get("t"), str):
        return None
    return {k: v for k, v in msg.items() if k not in FORBIDDEN_FIELDS}


def dumps(msg: dict[str, Any]) -> str:
    """Compact JSON for the wire (UTF-8 emoji kept as-is)."""
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":"))


def status_msg(state: str, msg: str = "") -> dict[str, Any]:
    return {"t": "status", "state": state, "msg": msg}
