"""MessageStore — Pichay's own compacted conversation history.

Instead of proxying Claude Code's full message array each turn,
the gateway maintains its own version where old tool results have
been replaced with tensor-handle stubs. Only this version is sent
to the upstream API.

Claude Code's message array is append-only: messages are never
modified, deleted, or reordered. We assert this invariant on each
ingest and log violations for debugging.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

from pichay.pager import PageStore, compact_messages

_DIM = "\033[2m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _strip_cache_control(msg: dict) -> None:
    """Remove cache_control from a message and its content blocks.

    Claude Code places cache_control markers for Anthropic's prompt
    caching. These don't apply to our compacted chain and accumulate
    past the API's 4-block limit.
    """
    msg.pop("cache_control", None)
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                block.pop("cache_control", None)


def _fingerprint(msg: dict) -> str:
    """Stable fingerprint for a message.

    Uses role + first 512 bytes of serialized content.  Tool results
    include tool_use_id which is unique; assistant/user messages use
    content prefix.  This is cheap and sufficient for detecting
    mutations — not a full content hash.
    """
    role = msg.get("role", "")
    # tool_use_id is the most stable identifier
    tool_use_id = msg.get("tool_use_id", "")
    if tool_use_id:
        return f"{role}:{tool_use_id}"
    # For content blocks, use first 512 bytes
    content = msg.get("content", "")
    if isinstance(content, list):
        raw = json.dumps(content, sort_keys=True, default=str)[:512]
    else:
        raw = str(content)[:512]
    h = hashlib.sha256(f"{role}:{raw}".encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{role}:{h}"


@dataclass
class IngestResult:
    """Result of ingesting a new turn's messages."""
    new_count: int = 0
    mutations_detected: int = 0
    deletions_detected: int = 0
    compacted_count: int = 0
    bytes_saved: int = 0


class MessageStore:
    """Pichay's compacted conversation history for a session."""

    def __init__(self, session_id: str, page_store: PageStore,
                 log_path: Path | None = None):
        self.session_id = session_id
        self.page_store = page_store
        self.log_path = log_path
        # Our compacted message list — this is what goes to the API
        self._messages: list[dict] = []
        # Fingerprints of known messages for append-only assertion
        self._fingerprints: list[str] = []
        # Stats
        self.total_ingested: int = 0
        self.total_mutations: int = 0
        self.total_deletions: int = 0
        self._turn: int = 0

    def _log_violation(self, kind: str, index: int, msg: dict | None,
                       expected_fp: str, actual_fp: str) -> None:
        """Log append-only violations to file for later analysis."""
        if self.log_path is None:
            return
        record = {
            "type": "append_only_violation",
            "kind": kind,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "turn": self._turn,
            "message_index": index,
            "expected_fingerprint": expected_fp,
            "actual_fingerprint": actual_fp,
        }
        if msg is not None:
            record["role"] = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                record["content_preview"] = json.dumps(content, default=str)[:500]
            else:
                record["content_preview"] = str(content)[:500]
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @property
    def messages(self) -> list[dict]:
        """The compacted message list to send to the API."""
        return self._messages

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def ingest(
        self,
        incoming: list[dict],
        age_threshold: int = 4,
        min_evict_size: int = 500,
    ) -> IngestResult:
        """Ingest a new turn's messages from Claude Code.

        Asserts append-only invariant on known messages, extracts
        new messages, compacts them, and appends to our store.

        Returns IngestResult with stats about what happened.
        """
        self._turn += 1
        result = IngestResult()
        known_count = len(self._fingerprints)

        # ── Assert append-only on known messages ─────────────────
        # Check that messages[0:known_count] haven't changed
        check_limit = min(known_count, len(incoming))
        for i in range(check_limit):
            fp = _fingerprint(incoming[i])
            if fp != self._fingerprints[i]:
                result.mutations_detected += 1
                self.total_mutations += 1
                self._log_violation(
                    "mutation", i, incoming[i],
                    self._fingerprints[i], fp,
                )
                print(
                    f"  {_YELLOW}[{self.session_id}] APPEND-ONLY VIOLATION at index {i}: "
                    f"expected {self._fingerprints[i][:32]}, "
                    f"got {fp[:32]}{_RESET}",
                    file=sys.stderr,
                )
                # Update our copy with the mutated version
                self._messages[i] = copy.deepcopy(incoming[i])
                self._fingerprints[i] = fp

        # Check for deletions (incoming shorter than known)
        if len(incoming) < known_count:
            deleted = known_count - len(incoming)
            result.deletions_detected = deleted
            self.total_deletions += deleted
            self._log_violation(
                "deletion", known_count, None,
                f"expected_{known_count}", f"got_{len(incoming)}",
            )
            print(
                f"  {_RED}[{self.session_id}] DELETION DETECTED: "
                f"expected {known_count} messages, got {len(incoming)} "
                f"({deleted} removed){_RESET}",
                file=sys.stderr,
            )
            # Truncate our store to match
            self._messages = self._messages[:len(incoming)]
            self._fingerprints = self._fingerprints[:len(incoming)]

        # ── Extract and append new messages ──────────────────────
        new_start = min(known_count, len(incoming))
        new_messages = incoming[new_start:]
        result.new_count = len(new_messages)

        if new_messages:
            # Deep copy new messages so we own them
            new_copies = copy.deepcopy(new_messages)

            # Strip cache_control — Claude Code's caching hints don't
            # apply to our compacted chain.  The API limits to 4 blocks
            # with cache_control; accumulated copies would overflow.
            for msg in new_copies:
                _strip_cache_control(msg)

            # Fingerprint before compaction (originals)
            for msg in new_messages:
                self._fingerprints.append(_fingerprint(msg))

            # Append to our store
            self._messages.extend(new_copies)
            self.total_ingested += len(new_copies)

        # ── Compact our entire store ─────────────────────────────
        # Age-based eviction runs on OUR messages, not Claude Code's
        compact_stats = compact_messages(
            self._messages,
            age_threshold=age_threshold,
            min_size=min_evict_size,
            page_store=self.page_store,
        )
        if compact_stats.evicted_count > 0:
            result.compacted_count = compact_stats.evicted_count
            result.bytes_saved = compact_stats.bytes_before - compact_stats.bytes_after

        return result
