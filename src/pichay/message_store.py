"""MessageStore — Pichay's gateway conversation store.

The gateway maintains its own physical message store, decoupled from
Claude Code's message array.  The physical store is what goes to the
upstream API.  Claude Code's mutations (system-reminder re-injections)
and deletions (compaction) are logged but do NOT propagate to the
physical store.

Architecture (the page table):
  - _messages: physical store (Pichay-owned, stable, sent to API)
  - _fingerprints: physical store fingerprints at ingest time
  - _client_fps: tracks what the client sent last turn (for mutation detection)
  - _client_to_physical: maps client indices to physical indices
    (diverges after client deletions)

Claude Code deletions are a no-op on the physical store.  The pager
manages eviction independently via compact_messages().  This eliminates
the "double KV cache tax" where client compaction and pager eviction
each independently invalidated the API-side cache prefix.
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
        # Physical store — Pichay-owned, sent to API, stable
        self._messages: list[dict] = []
        # Fingerprints of physical messages at ingest time
        self._fingerprints: list[str] = []
        # Client tracking — separate from physical store
        self._client_fps: list[str] = []  # client's current fingerprints
        self._client_to_physical: list[int] = []  # client idx -> physical idx
        # Stats
        self.total_ingested: int = 0
        self.total_mutations: int = 0
        self.total_deletions: int = 0
        self.total_client_deletions_absorbed: int = 0
        self._turn: int = 0

    @staticmethod
    def _content_size(msg: dict) -> int:
        """Approximate byte size of a message's content."""
        content = msg.get("content", "")
        if isinstance(content, list):
            return len(json.dumps(content, default=str))
        return len(str(content))

    @staticmethod
    def _content_preview(msg: dict, limit: int = 500) -> str:
        """Extract a content preview string from a message."""
        content = msg.get("content", "")
        if isinstance(content, list):
            return json.dumps(content, default=str)[:limit]
        return str(content)[:limit]

    def _log_violation(self, kind: str, index: int, msg: dict | None,
                       expected_fp: str, actual_fp: str,
                       old_msg: dict | None = None,
                       deleted_msgs: list[dict] | None = None) -> None:
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
            record["new_size"] = self._content_size(msg)
            record["new_preview"] = self._content_preview(msg)
        if old_msg is not None:
            record["old_size"] = self._content_size(old_msg)
            record["old_preview"] = self._content_preview(old_msg)
        if deleted_msgs:
            record["deleted_count"] = len(deleted_msgs)
            record["deleted_messages"] = [
                {
                    "index": index - len(deleted_msgs) + i,
                    "role": m.get("role", ""),
                    "size": self._content_size(m),
                    "preview": self._content_preview(m, limit=200),
                }
                for i, m in enumerate(deleted_msgs)
            ]
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

        Client tracking (fingerprints, index mapping) is separate from
        the physical store.  Mutations and deletions update client state
        only — the physical store stays stable for KV cache coherence.

        Returns IngestResult with stats about what happened.
        """
        self._turn += 1
        result = IngestResult()
        client_known = len(self._client_fps)

        # ── Detect mutations in known client messages ────────────
        check_limit = min(client_known, len(incoming))
        for i in range(check_limit):
            fp = _fingerprint(incoming[i])
            if fp != self._client_fps[i]:
                result.mutations_detected += 1
                self.total_mutations += 1
                # Look up physical message via mapping for comparison
                phys_idx = self._client_to_physical[i]
                old_msg = self._messages[phys_idx] if phys_idx < len(self._messages) else None
                self._log_violation(
                    "mutation", i, incoming[i],
                    self._client_fps[i], fp,
                    old_msg=old_msg,
                )
                print(
                    f"  {_YELLOW}[{self.session_id}] APPEND-ONLY VIOLATION at index {i}: "
                    f"expected {self._client_fps[i][:32]}, "
                    f"got {fp[:32]}{_RESET}",
                    file=sys.stderr,
                )
                # Update CLIENT fingerprint only — physical store unchanged
                self._client_fps[i] = fp

        # ── Detect client deletions (compaction) ─────────────────
        if len(incoming) < client_known:
            deleted = client_known - len(incoming)
            result.deletions_detected = deleted
            self.total_deletions += deleted
            self.total_client_deletions_absorbed += deleted
            # Log what the client is deleting (from physical store via mapping)
            deleted_physical = []
            for ci in range(len(incoming), client_known):
                pi = self._client_to_physical[ci]
                if pi < len(self._messages):
                    deleted_physical.append(self._messages[pi])
            self._log_violation(
                "deletion", client_known, None,
                f"expected_{client_known}", f"got_{len(incoming)}",
                deleted_msgs=deleted_physical,
            )
            print(
                f"  {_DIM}[{self.session_id}] CLIENT DELETION ABSORBED: "
                f"{deleted} messages dropped by client, "
                f"physical store unchanged ({len(self._messages)} msgs){_RESET}",
                file=sys.stderr,
            )
            # Truncate CLIENT tracking only — physical store stays intact
            self._client_fps = self._client_fps[:len(incoming)]
            self._client_to_physical = self._client_to_physical[:len(incoming)]

        # ── Extract and append new messages ──────────────────────
        new_start = min(client_known, len(incoming))
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

            # Track in physical store and client mapping
            phys_start = len(self._messages)
            for j, msg in enumerate(new_messages):
                fp = _fingerprint(msg)
                self._fingerprints.append(fp)
                self._client_fps.append(fp)
                self._client_to_physical.append(phys_start + j)

            # Append to physical store
            self._messages.extend(new_copies)
            self.total_ingested += len(new_copies)

        # ── Compact physical store ───────────────────────────────
        # Age-based eviction runs on OUR messages, not Claude Code's.
        # This is the only thing that modifies the physical store.
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
