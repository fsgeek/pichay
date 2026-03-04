"""Conversation block store: content-addressed KV for message blocks.

Assigns short stable IDs to conversation message blocks based on
content hashing. The model sees these labels and can reference them
in cleanup operations (Phase 2).

Phase 1: labeling only — no cleanup tag processing yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class BlockEntry:
    """A tracked conversation block."""
    block_id: str           # Short hex ID (first 8 chars of content hash)
    content_hash: str       # Full SHA-256 of content
    size: int               # Byte size of original content
    turn: int               # Turn when first seen
    role: str               # "user" or "assistant"
    preview: str            # First 80 chars for logging
    status: str = "resident"  # resident | anchored | summarized | dropped
    original_content: str | None = None  # Full content for fault restoration
    summary: str | None = None  # Model-authored summary (if summarized)


class BlockStore:
    """Per-session content-addressed store for conversation blocks.

    Maps content hashes to BlockEntries. Assigns short IDs for model
    reference. Detects content changes (re-labels gracefully).
    """

    def __init__(self):
        self._by_id: dict[str, BlockEntry] = {}
        self._by_hash: dict[str, str] = {}  # content_hash → block_id

    def label_messages(self, messages: list[dict], current_turn: int) -> None:
        """Inject [block:xxxx] labels into message content.

        Modifies messages in-place. Each message gets a label based on
        its content hash. Labels are stable across turns as long as
        the content doesn't change.

        Only labels user and assistant text messages. Tool_use and
        tool_result blocks are managed by the PageStore, not here.
        """
        for msg in messages:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")

            # String content (simple user messages)
            if isinstance(content, str):
                # Skip if already labeled
                if content.startswith("[block:"):
                    continue
                # Skip very short messages (not worth labeling)
                if len(content) < 200:
                    continue

                entry = self._get_or_create(content, role, current_turn)
                if entry and entry.status == "resident":
                    msg["content"] = f"[block:{entry.block_id}]\n{content}"

            # List content (structured blocks)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "text":
                        continue
                    text = block.get("text", "")
                    # Skip if already labeled
                    if text.startswith("[block:"):
                        continue
                    # Skip short blocks
                    if len(text) < 200:
                        continue

                    entry = self._get_or_create(text, role, current_turn)
                    if entry and entry.status == "resident":
                        block["text"] = f"[block:{entry.block_id}]\n{text}"

    def _get_or_create(self, content: str, role: str,
                       turn: int) -> BlockEntry | None:
        """Get existing entry by content hash, or create a new one."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        short_id = content_hash[:8]

        # Already tracked
        if content_hash in self._by_hash:
            return self._by_id[self._by_hash[content_hash]]

        # Check for ID collision (different content, same short hash)
        if short_id in self._by_id:
            existing = self._by_id[short_id]
            if existing.content_hash != content_hash:
                # Collision — extend the ID
                short_id = content_hash[:12]
                if short_id in self._by_id:
                    return None  # Extremely unlikely, skip labeling

        preview = content[:80].replace("\n", " ")
        entry = BlockEntry(
            block_id=short_id,
            content_hash=content_hash,
            size=len(content.encode()),
            turn=turn,
            role=role,
            preview=preview,
            original_content=content,
        )
        self._by_id[short_id] = entry
        self._by_hash[content_hash] = short_id
        return entry

    def get(self, block_id: str) -> BlockEntry | None:
        """Look up a block by its short ID."""
        return self._by_id.get(block_id)

    def restore(self, block_id: str) -> str | None:
        """Restore a compressed block's original content."""
        entry = self._by_id.get(block_id)
        if entry and entry.original_content:
            entry.status = "resident"
            return entry.original_content
        return None

    @property
    def block_count(self) -> int:
        return len(self._by_id)

    @property
    def total_bytes(self) -> int:
        return sum(e.size for e in self._by_id.values()
                   if e.status == "resident")

    def large_blocks(self, min_size: int = 2000) -> list[BlockEntry]:
        """Return resident blocks larger than min_size, sorted by size."""
        return sorted(
            [e for e in self._by_id.values()
             if e.status == "resident" and e.size >= min_size],
            key=lambda e: e.size,
            reverse=True,
        )

    def summary(self) -> dict:
        """Return a summary for the health endpoint."""
        by_status = {}
        for e in self._by_id.values():
            by_status[e.status] = by_status.get(e.status, 0) + 1
        return {
            "total_blocks": len(self._by_id),
            "total_bytes": self.total_bytes,
            "by_status": by_status,
        }
