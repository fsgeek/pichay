from __future__ import annotations

import re


def parse_duration(value: str) -> int:
    """Parse duration strings like 24h, 15m, 7d into seconds."""
    m = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", value)
    if not m:
        raise ValueError(f"invalid duration: {value!r}")
    n = int(m.group(1))
    unit = m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return n * mult


def content_bytes(content: object) -> int:
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    if isinstance(content, list):
        return sum(content_bytes(x) for x in content)
    if isinstance(content, dict):
        return sum(content_bytes(v) for v in content.values())
    return len(str(content).encode("utf-8"))
