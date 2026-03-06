"""Conversation block store: content-addressed KV for message blocks.

Assigns short stable IDs to conversation message blocks based on
content hashing. The model sees these labels and can reference them
in cleanup operations.

Phase 1: labeling — inject [block:xxxx] markers into message content.
Phase 2: cleanup — drop, summarize, anchor blocks via inline tags.
"""

from __future__ import annotations

import hashlib
import re
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
                if content.startswith("[tensor:") or content.startswith("[block:"):
                    continue
                # Skip very short messages (not worth labeling)
                if len(content) < 200:
                    continue

                entry = self._get_or_create(content, role, current_turn)
                if entry and entry.status == "resident":
                    size_k = entry.size / 1024
                    msg["content"] = f"[tensor:{entry.block_id} ({size_k:.1f}KB)]\n{content}"

            # List content (structured blocks)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "text":
                        continue
                    text = block.get("text", "")
                    # Skip if already labeled
                    if text.startswith("[tensor:") or text.startswith("[block:"):
                        continue
                    # Skip short blocks
                    if len(text) < 200:
                        continue

                    entry = self._get_or_create(text, role, current_turn)
                    if entry and entry.status == "resident":
                        size_k = entry.size / 1024
                        block["text"] = f"[tensor:{entry.block_id} ({size_k:.1f}KB)]\n{text}"

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

    # --- Phase 2: Cleanup operations ---

    def drop(self, block_id: str) -> bool:
        """Mark a block as dropped. Content stays for audit but won't
        appear in future messages."""
        entry = self._by_id.get(block_id)
        if not entry:
            return False
        entry.status = "dropped"
        return True

    def summarize(self, block_id: str, summary: str) -> bool:
        """Replace a block's content with a model-authored summary."""
        entry = self._by_id.get(block_id)
        if not entry:
            return False
        entry.status = "summarized"
        entry.summary = summary
        return True

    def anchor(self, block_id: str) -> bool:
        """Mark a block for retention — hint to keep in working memory."""
        entry = self._by_id.get(block_id)
        if not entry:
            return False
        entry.status = "anchored"
        return True

    _BLOCK_LABEL_RE = re.compile(r"^\[(?:tensor|block):([a-f0-9]{8,12})(?:\s*\([^)]*\))?\]\n?")

    def apply_to_messages(self, messages: list[dict]) -> dict:
        """Apply block status to messages — replace dropped/summarized content.

        Modifies messages in-place. Returns stats dict.
        """
        stats = {"dropped": 0, "summarized": 0, "anchored": 0}

        for msg in messages:
            content = msg.get("content", "")

            if isinstance(content, str):
                msg["content"] = self._apply_to_text(content, msg, stats)

            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "text":
                        continue
                    text = block.get("text", "")
                    block["text"] = self._apply_to_text(text, msg, stats)

        return stats

    def _apply_to_text(self, text: str, msg: dict, stats: dict) -> str:
        """Apply block status to a single text content."""
        m = self._BLOCK_LABEL_RE.match(text)
        if not m:
            return text

        block_id = m.group(1)
        entry = self._by_id.get(block_id)
        if not entry:
            return text

        if entry.status == "dropped":
            stats["dropped"] += 1
            turn_info = f"message {entry.turn} in session log"
            return (
                f"[...archived {entry.size:,} chars, {turn_info}...]"
            )

        if entry.status == "summarized" and entry.summary:
            stats["summarized"] += 1
            return (
                f"[tensor:{block_id} — summarized, was {entry.size:,} chars]\n"
                f"{entry.summary}"
            )

        # resident or anchored — no change
        if entry.status == "anchored":
            stats["anchored"] += 1

        return text

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
