from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CanonicalMessage:
    role: str
    content: list[dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalRequest:
    provider: str
    model: str
    max_tokens: int | None
    stream: bool
    messages: list[CanonicalMessage]
    tools: list[dict[str, Any]] = field(default_factory=list)
    system: Any = None
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyAction:
    stage: str
    action: str
    target_id: str
    message_index: int
    block_index: int
    replacement_text: str | None
    bytes: int
    duplication_score: float


@dataclass
class PolicyContext:
    protected_targets: set[str] = field(default_factory=set)
