"""Conversation block store: content-addressed KV for message blocks.

Assigns short stable IDs to conversation message blocks based on
content hashing. The model sees these labels and can reference them
in cleanup operations.

Phase 1: labeling — inject [block:xxxx] markers into message content.
Phase 2: cleanup — drop, summarize, anchor blocks via inline tags.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


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

    _LABEL_RE = re.compile(r"^\[(?:tensor|block):([0-9a-f]+)[ (]")

    def _has_our_label(self, text: str) -> bool:
        """Check if text starts with a label WE placed.

        Validates the ID against known block IDs. Foreign labels
        (injected via file contents, tool results, or crafted messages)
        will have unrecognized IDs and return False, preventing an
        attacker from skipping labeling or spoofing block references.
        """
        m = self._LABEL_RE.match(text)
        if m is None:
            return False
        return m.group(1) in self._by_id

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
                # Skip if already labeled by us (validated against known IDs)
                if self._has_our_label(content):
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
                    # Skip if already labeled by us (validated against known IDs)
                    if self._has_our_label(text):
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

    def collapse_range(self, start_turn: int, end_turn: int,
                       summary: str) -> list[str]:
        """Replace all blocks in a turn range with a summary marker.

        Marks all resident/anchored blocks in [start_turn, end_turn] as
        dropped, then creates a synthetic summary block covering the range.
        Returns list of block IDs that were collapsed.

        The synthetic block gets a deterministic ID from the range + summary
        so repeated collapse of the same range is idempotent.
        """
        collapsed_ids = []
        for entry in self._by_id.values():
            if (start_turn <= entry.turn <= end_turn
                    and entry.status in ("resident", "anchored")):
                entry.status = "dropped"
                collapsed_ids.append(entry.block_id)

        if not collapsed_ids:
            return []

        # Create a synthetic summary block for the range
        synthetic_content = (
            f"[Turns {start_turn}-{end_turn} collapsed: {summary}]"
        )
        content_hash = hashlib.sha256(synthetic_content.encode()).hexdigest()
        short_id = content_hash[:8]

        # Avoid collision with existing blocks
        if short_id in self._by_id:
            short_id = content_hash[:12]

        if short_id not in self._by_id:
            entry = BlockEntry(
                block_id=short_id,
                content_hash=content_hash,
                size=len(synthetic_content.encode()),
                turn=start_turn,
                role="assistant",
                preview=synthetic_content[:80],
                status="summarized",
                summary=summary,
            )
            self._by_id[short_id] = entry
            self._by_hash[content_hash] = short_id

        return collapsed_ids

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

    # --- Checkpoint / Restart ---

    def checkpoint(self, path: Path) -> int:
        """Serialize block state to JSON. Returns count of entries saved.

        Writes atomically (tmp + rename) so a crash mid-write can't
        corrupt the checkpoint. Only saves metadata and summaries —
        original_content is NOT checkpointed (too large, and the
        messages array is the source of truth for content).
        """
        entries = []
        for entry in self._by_id.values():
            entries.append({
                "block_id": entry.block_id,
                "content_hash": entry.content_hash,
                "size": entry.size,
                "turn": entry.turn,
                "role": entry.role,
                "preview": entry.preview,
                "status": entry.status,
                "summary": entry.summary,
            })

        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2))
        tmp.rename(path)
        return len(entries)

    @classmethod
    def from_checkpoint(cls, path: Path) -> "BlockStore":
        """Restore a BlockStore from a checkpoint file.

        Returns a new BlockStore with all tracked entries restored.
        Blocks that were resident at checkpoint time stay resident
        but without original_content (will be re-populated when
        label_messages sees the same content again).
        """
        store = cls()
        if not path.is_file():
            return store

        try:
            entries = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return store

        for rec in entries:
            entry = BlockEntry(
                block_id=rec["block_id"],
                content_hash=rec["content_hash"],
                size=rec["size"],
                turn=rec["turn"],
                role=rec["role"],
                preview=rec["preview"],
                status=rec.get("status", "resident"),
                summary=rec.get("summary"),
                original_content=None,
            )
            store._by_id[entry.block_id] = entry
            store._by_hash[entry.content_hash] = entry.block_id

        return store
