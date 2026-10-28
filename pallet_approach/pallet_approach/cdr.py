"""Rewrite a serialized message's header.frame_id without decoding it (D101 relay).

CDR layout of a header: bytes 0-3 encapsulation (byte 1: 1 = little-endian),
4-11 the stamp, 12-15 the frame id's length (with its NUL), then the string,
padded to 4 bytes (alignment counts from byte 4). Re-padding shifts the rest
by a multiple of 4, which is safe for messages with no 8-byte fields, such as
sensor_msgs/PointCloud2.
"""

from __future__ import annotations

import struct


def _pad(n: int) -> int:
    return (4 - n % 4) % 4


def restamp(raw: bytes, frame_id: str) -> bytes:
    if len(raw) < 16:
        raise ValueError('too short to hold a header')
    little = raw[1] == 0x01
    fmt = '<I' if little else '>I'
    old_len = struct.unpack_from(fmt, raw, 12)[0]
    old_end = 16 + old_len
    old_end += _pad(old_end - 4)
    name = frame_id.encode() + b'\x00'
    head = raw[:12] + struct.pack(fmt, len(name)) + name
    head += b'\x00' * _pad(len(head) - 4)
    return head + raw[old_end:]


def frame_id(raw: bytes) -> str:
    """The header.frame_id of a serialized message (for checks and logs)."""
    fmt = '<I' if raw[1] == 0x01 else '>I'
    n = struct.unpack_from(fmt, raw, 12)[0]
    return raw[16:16 + n - 1].decode()
