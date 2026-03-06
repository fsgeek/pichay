"""Context window pager — evicts stale tool results from the messages array.

This is the intervention layer for the phase 1 context utilization experiment.
It sits between Claude Code and Anthropic's API (inside the proxy) and replaces
old, large tool results with compact summaries. The originals are stored in a
page file for logging and analysis.

Design decisions:
- FIFO eviction: oldest results first (data shows Q1 results have 0.896
  amplification ratio — evicting them captures the most benefit)
- No recall tool injection: if the model needs evicted content, it already
  knows how to re-issue the tool call (Read, Grep, etc.). The "page fault"
  is just a new tool call. PDP-11 overlays, not virtual memory.
- Error results are never evicted (the model needs those for debugging)
- Small results (<min_size bytes) aren't worth compacting

Metrics for success (from Tony):
- Lower token consumption
- Better quality output (fewer tokens → LLMs work better)
- Faster responses
- Slower consumption of the context window
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PageEntry:
    """A single evicted tool result."""

    tool_use_id: str
    tool_name: str
    tool_input: dict
    original_content: str | list
    original_size: int
    summary: str
    evicted_at: float  # time.monotonic()
    turn_index: int
    turns_from_end: int


@dataclass
class PageFault:
    """A tool call that re-requests evicted content."""

    tool_use_id: str
    tool_name: str
    tool_input: dict
    original_eviction: PageEntry
    detected_at: float  # time.monotonic()


@dataclass
class CompactionStats:
    """What happened during a single compaction pass."""

    total_tool_results: int = 0
    evicted_count: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    skipped_small: int = 0
    skipped_recent: int = 0
    skipped_error: int = 0
    skipped_pinned: int = 0

    @property
    def bytes_saved(self) -> int:
        return self.bytes_before - self.bytes_after

    @property
    def reduction_pct(self) -> float:
        if self.bytes_before == 0:
            return 0.0
        return (self.bytes_saved / self.bytes_before) * 100


class PageStore:
    """Stores evicted tool result content and detects page faults.

    Distinguishes two operations the proxy performs:

    - **Eviction**: Removing Read results (stable content identity).
      Tracked in the eviction index. Faults are possible.
    - **Garbage collection**: Removing ephemeral tool output (Bash,
      Grep, Glob, etc.). Always safe — re-running requests current
      state, not the evicted content. No fault concept.

    Re-eviction (same tool_use_id seen on a subsequent turn) is a
    no-op for counting — Claude Code re-sends its full message history,
    so the proxy re-stubs the same content every turn. That's the
    mechanical operation, not a new eviction decision.
    """

    def __init__(self, log_path: Path | None = None):
        self.pages: dict[str, PageEntry] = {}
        self.log_path = log_path
        self.faults: list[PageFault] = []
        # Eviction index: file_path → PageEntry (Read only)
        self._eviction_index: dict[str, PageEntry] = {}

        # Split counters — eviction vs garbage collection
        self.unique_evictions: int = 0
        self.eviction_bytes_saved: int = 0
        self.gc_count: int = 0
        self.gc_bytes_saved: int = 0

        # Fault-driven pinning: one fault + same content = pin
        self._pinned: dict[str, str] = {}  # file_path → content_hash
        self._fault_content: dict[str, str] = {}  # file_path → content_hash at eviction
        self.pin_count: int = 0

        # Model-initiated release: paths the model says it's done with
        self._released: set[str] = set()
        self.release_count: int = 0

        # Tensor index: tensor_handle → PageEntry (unified addressing)
        self._tensor_index: dict[str, PageEntry] = {}

        # Legacy aliases for compatibility
        self.cumulative_evictions: int = 0
        self.cumulative_bytes_saved: int = 0

    def store(self, entry: PageEntry) -> str | None:
        """Store an evicted entry. Returns tensor handle for new evictions."""
        is_new = entry.tool_use_id not in self.pages
        self.pages[entry.tool_use_id] = entry

        if not is_new:
            return None  # Re-eviction — same content re-stubbed, don't count

        # Generate tensor handle from content hash
        content_str = (entry.original_content
                       if isinstance(entry.original_content, str)
                       else json.dumps(entry.original_content))
        tensor_handle = hashlib.sha256(
            content_str.encode("utf-8")
        ).hexdigest()[:8]
        self._tensor_index[tensor_handle] = entry

        bytes_saved = entry.original_size - len(
            entry.summary.encode("utf-8")
        )
        key = _eviction_key(entry.tool_name, entry.tool_input)

        if key is not None:
            # Read result — real eviction, track in index
            self._eviction_index[key] = entry
            self.unique_evictions += 1
            self.eviction_bytes_saved += bytes_saved
        else:
            # Ephemeral tool — garbage collection
            self.gc_count += 1
            self.gc_bytes_saved += bytes_saved

        # Update legacy counters
        self.cumulative_evictions = self.unique_evictions + self.gc_count
        self.cumulative_bytes_saved = (
            self.eviction_bytes_saved + self.gc_bytes_saved
        )

        if self.log_path is not None:
            record = {
                "type": "eviction",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool_use_id": entry.tool_use_id,
                "tool_name": entry.tool_name,
                "tool_category": "eviction" if key else "gc",
                "original_size": entry.original_size,
                "summary_size": len(entry.summary.encode("utf-8")),
                "turn_index": entry.turn_index,
                "turns_from_end": entry.turns_from_end,
                "tensor_handle": tensor_handle,
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return tensor_handle

    def resolve_tensor(self, handle: str) -> PageEntry | None:
        """Resolve a tensor handle to its PageEntry."""
        return self._tensor_index.get(handle)

    def detect_faults(self, messages: list[dict]) -> list[PageFault]:
        """Scan recent tool_use blocks for re-requests of evicted content.

        Looks at tool_use blocks in assistant messages and checks if any
        match evicted pages. A match means the model needed content we
        took away — a page fault.
        """
        new_faults = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue

                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                tool_use_id = block.get("id", "")

                # Skip the original tool_use that produced the
                # evicted result — that's not a fault, it's history
                if tool_use_id in self.pages:
                    continue

                key = _eviction_key(tool_name, tool_input)
                if key is None:
                    continue

                evicted = self._eviction_index.get(key)
                if evicted is None:
                    continue

                # Don't double-count: check if this tool_use_id
                # was already recorded as a fault
                if any(f.tool_use_id == tool_use_id for f in self.faults):
                    continue

                fault = PageFault(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    original_eviction=evicted,
                    detected_at=time.monotonic(),
                )
                new_faults.append(fault)
                self.faults.append(fault)

                # Record evicted content hash for pin comparison
                self._fault_content[key] = _content_hash(
                    evicted.original_content
                )

                if self.log_path is not None:
                    record = {
                        "type": "page_fault",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "tool_name": tool_name,
                        "eviction_key": key,
                        "original_tool_use_id": evicted.tool_use_id,
                        "original_size": evicted.original_size,
                        "original_turn": evicted.turn_index,
                    }
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

        return new_faults

    def is_pinned(self, key: str) -> bool:
        return key in self._pinned

    def pin(self, key: str, content_hash: str) -> None:
        self._pinned[key] = content_hash
        self._fault_content.pop(key, None)
        self.pin_count += 1

        if self.log_path is not None:
            record = {
                "type": "pin",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "file_path": key,
                "content_hash": content_hash,
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def unpin(self, key: str) -> None:
        old_hash = self._pinned.pop(key, None)
        self._fault_content.pop(key, None)

        if self.log_path is not None and old_hash is not None:
            record = {
                "type": "unpin",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "file_path": key,
                "reason": "content_changed",
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def retrieve(self, tool_use_id: str) -> PageEntry | None:
        return self.pages.get(tool_use_id)

    @property
    def fault_rate(self) -> float:
        """Fault rate over unique Read evictions (the meaningful denominator)."""
        if self.unique_evictions == 0:
            return 0.0
        return len(self.faults) / self.unique_evictions

    @property
    def total_bytes_saved(self) -> int:
        return self.eviction_bytes_saved + self.gc_bytes_saved

    def summary(self) -> dict:
        """Current state for the health endpoint."""
        return {
            "unique_evictions": self.unique_evictions,
            "gc_count": self.gc_count,
            "total_bytes_saved": self.total_bytes_saved,
            "eviction_bytes_saved": self.eviction_bytes_saved,
            "gc_bytes_saved": self.gc_bytes_saved,
            "total_page_faults": len(self.faults),
            "fault_rate": self.fault_rate,
            "pages_in_store": len(self.pages),
            "pinned_count": len(self._pinned),
            "pinned_paths": list(self._pinned.keys()),
            "total_pins": self.pin_count,
            "faults_by_tool": _count_by(
                self.faults, lambda f: f.tool_name
            ),
            "evictions_by_tool": _count_by(
                list(self.pages.values()), lambda p: p.tool_name
            ),
        }


def _count_by(items: list, key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _eviction_key(tool_name: str, tool_input: dict) -> str | None:
    """Extract the identifying key for a tool call, for fault matching.

    Only Read produces a fault-trackable key. Read results have stable
    content identity (file path) — re-requesting after eviction means
    the model needed what was taken.

    Bash, Grep, Glob, WebFetch, WebSearch are ephemeral — re-running
    them requests current state, not the evicted content. Removing
    their output is garbage collection, not eviction. No fault possible.
    """
    if tool_name == "Read":
        return tool_input.get("file_path")
    return None


def _content_size(content: str | list | None) -> int:
    """Measure content size in bytes."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    return len(json.dumps(content).encode("utf-8"))


def _content_hash(content: str | list | None) -> str:
    """Stable hash of tool result content for pin comparison."""
    if content is None:
        raw = b""
    elif isinstance(content, str):
        raw = content.encode("utf-8")
    else:
        raw = json.dumps(content, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _build_tool_use_index(messages: list[dict]) -> dict[str, dict]:
    """Map tool_use_id → {name, input} from assistant messages."""
    index = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                index[block["id"]] = {
                    "name": block.get("name", "unknown"),
                    "input": block.get("input", {}),
                }
    return index


def _make_summary(tool_name: str, tool_input: dict, original_size: int,
                  original_content: str | list | None = None,
                  tool_use_id: str | None = None,
                  tensor_handle: str | None = None) -> str:
    """Generate a compact summary for an evicted tool result.

    The summary tells the model what WAS here via a tensor handle.
    All evicted content uses the same format: [tensor:handle — description].
    The model recalls any tensor via the recall tool using the handle.
    """
    size_str = f"{original_size:,}"
    handle = tensor_handle or "unknown"

    if tool_name == "Read":
        path = tool_input.get("file_path", "unknown")
        line_count = ""
        if isinstance(original_content, str):
            lines = original_content.count("\n")
            line_count = f", {lines} lines"
        return (
            f"[tensor:{handle} — {path} ({size_str} bytes{line_count})]"
        )

    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "?")
        path = tool_input.get("path", ".")
        match_info = ""
        if isinstance(original_content, str):
            result_lines = len(original_content.strip().splitlines())
            match_info = f", {result_lines} results"
        return (
            f"[tensor:{handle} — Grep '{pattern}' in {path}"
            f" ({size_str} bytes{match_info})]"
        )

    elif tool_name == "Glob":
        pattern = tool_input.get("pattern", "?")
        match_info = ""
        if isinstance(original_content, str):
            matches = len(original_content.strip().splitlines())
            match_info = f", {matches} matches"
        return (
            f"[tensor:{handle} — Glob '{pattern}'"
            f" ({size_str} bytes{match_info})]"
        )

    elif tool_name == "Bash":
        cmd = tool_input.get("command", "?")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        return (
            f"[tensor:{handle} — Bash `{cmd}` ({size_str} bytes)]"
        )

    elif tool_name == "WebFetch":
        url = tool_input.get("url", "?")
        return (
            f"[tensor:{handle} — WebFetch {url} ({size_str} bytes)]"
        )

    elif tool_name == "WebSearch":
        query = tool_input.get("query", "?")
        return (
            f"[tensor:{handle} — WebSearch '{query}' ({size_str} bytes)]"
        )

    else:
        return (
            f"[tensor:{handle} — {tool_name} ({size_str} bytes)]"
        )


def compact_messages(
    messages: list[dict],
    age_threshold: int = 4,
    min_size: int = 500,
    page_store: PageStore | None = None,
) -> CompactionStats:
    """Replace old, large tool results with compact summaries.

    Mutates the messages list in place. Returns stats about what was done.

    Args:
        messages: The messages array from the API request body.
        age_threshold: Results older than this many user-turns from the
            end get evicted. Default 4.
        min_size: Results smaller than this (bytes) are kept as-is.
            Not worth compacting a 22-byte Edit confirmation. Default 500.
        page_store: If provided, evicted content is stored here.
    """
    stats = CompactionStats()
    tool_use_index = _build_tool_use_index(messages)

    # Count user turns (each user message is a "turn")
    user_turn_indices: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            user_turn_indices.append(i)

    total_user_turns = len(user_turn_indices)

    # Phase 0: Check for fresh reads that should unpin stale pins
    if page_store is not None and page_store._pinned:
        _check_pin_freshness(messages, tool_use_index, page_store)

    # Walk through messages and compact old tool results
    current_user_turn = 0
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue

        current_user_turn += 1
        turns_from_end = total_user_turns - current_user_turn

        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for block_idx, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue

            stats.total_tool_results += 1
            original_content = block.get("content", "")
            original_size = _content_size(original_content)

            # Skip errors — model needs those
            if block.get("is_error", False):
                stats.skipped_error += 1
                continue

            # Resolve tool identity early for release check
            tool_use_id = block.get("tool_use_id", "")
            tool_info = tool_use_index.get(tool_use_id, {})
            tool_name = tool_info.get("name", "unknown")
            tool_input = tool_info.get("input", {})
            key = _eviction_key(tool_name, tool_input)

            # Model-initiated release: skip age check, evict immediately
            released = (
                key is not None
                and page_store is not None
                and key in page_store._released
            )

            # Skip recent results (unless released by the model)
            if not released and turns_from_end < age_threshold:
                stats.skipped_recent += 1
                stats.bytes_before += original_size
                stats.bytes_after += original_size
                continue

            # Skip small results (unless released)
            if not released and original_size < min_size:
                stats.skipped_small += 1
                stats.bytes_before += original_size
                stats.bytes_after += original_size
                continue

            # Skip pinned content — fault history proved it's in the working set
            if key is not None and page_store is not None:
                if page_store.is_pinned(key):
                    stats.skipped_pinned += 1
                    stats.bytes_before += original_size
                    stats.bytes_after += original_size
                    continue

                # Check if this should be pinned: faulted + same content
                if key in page_store._fault_content:
                    ch = _content_hash(original_content)
                    if page_store._fault_content[key] == ch:
                        page_store.pin(key, ch)
                        stats.skipped_pinned += 1
                        stats.bytes_before += original_size
                        stats.bytes_after += original_size
                        continue

            # This result gets evicted — store first to get tensor handle
            tensor_handle = None
            if page_store is not None:
                # Pre-compute summary placeholder for PageEntry
                entry = PageEntry(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    original_content=original_content,
                    original_size=original_size,
                    summary="",  # Will be updated below
                    evicted_at=time.monotonic(),
                    turn_index=current_user_turn,
                    turns_from_end=turns_from_end,
                )
                tensor_handle = page_store.store(entry)

            summary = _make_summary(
                tool_name, tool_input, original_size, original_content,
                tool_use_id=tool_use_id,
                tensor_handle=tensor_handle,
            )

            # Update the stored entry's summary
            if page_store is not None and tool_use_id in page_store.pages:
                page_store.pages[tool_use_id].summary = summary

            # Replace the content
            content[block_idx] = {**block, "content": summary}

            # Clear release flag after eviction
            if released and page_store is not None:
                page_store._released.discard(key)
                page_store.release_count += 1

            stats.evicted_count += 1
            stats.bytes_before += original_size
            stats.bytes_after += len(summary.encode("utf-8"))

    return stats


@dataclass
class ConversationCompactionStats:
    """What happened during conversation compression."""

    messages_scanned: int = 0
    messages_compressed: int = 0
    chars_before: int = 0
    chars_after: int = 0

    @property
    def chars_saved(self) -> int:
        return self.chars_before - self.chars_after


def compact_conversation(
    messages: list[dict],
    *,
    preserve_recent: int = 6,
    min_text_chars: int = 2000,
    max_compressed_chars: int = 200,
) -> ConversationCompactionStats:
    """Compress old conversation text in user and assistant messages.

    Replaces large text blocks in messages older than `preserve_recent`
    turns from the end with truncated versions plus retrieval handles.
    Tool results and tool_use blocks are untouched (handled by
    compact_messages). Only plain text content is compressed.

    The full conversation is preserved in the proxy's JSONL log.

    Args:
        messages: The messages array (modified in place).
        preserve_recent: Number of recent message pairs to keep intact.
        min_text_chars: Minimum text size to consider for compression.
        max_compressed_chars: Target size for compressed text.
    """
    stats = ConversationCompactionStats()

    # Count user messages to determine which are "old"
    total_messages = len(messages)
    if total_messages <= preserve_recent * 2:
        return stats  # Not enough messages to compress

    cutoff = total_messages - (preserve_recent * 2)

    for i, msg in enumerate(messages):
        if i >= cutoff:
            break  # Recent messages — preserve

        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue

        content = msg.get("content", "")
        stats.messages_scanned += 1

        # Handle string content (simple text messages)
        if isinstance(content, str):
            if len(content) < min_text_chars:
                continue
            stats.chars_before += len(content)
            # Keep first and last portions, add retrieval handle
            head = content[:max_compressed_chars // 2]
            tail = content[-(max_compressed_chars // 2):]
            compressed = (
                f"{head}\n[...archived {len(content):,} chars, "
                f"message {i} in session log...]\n{tail}"
            )
            msg["content"] = compressed
            stats.messages_compressed += 1
            stats.chars_after += len(compressed)
            continue

        # Handle list content (structured blocks)
        if not isinstance(content, list):
            continue

        for block_idx, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            # Only compress text blocks — leave tool_result and tool_use alone
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            if len(text) < min_text_chars:
                continue

            stats.chars_before += len(text)
            head = text[:max_compressed_chars // 2]
            tail = text[-(max_compressed_chars // 2):]
            compressed = (
                f"{head}\n[...archived {len(text):,} chars, "
                f"message {i} in session log...]\n{tail}"
            )
            content[block_idx] = {**block, "text": compressed}
            stats.messages_compressed += 1
            stats.chars_after += len(compressed)

    return stats


def _check_pin_freshness(
    messages: list[dict],
    tool_use_index: dict[str, dict],
    page_store: PageStore,
) -> None:
    """Detect fresh reads of pinned paths — unpin if content changed.

    When a file is pinned but the model reads it again (edit/review cycle),
    the new read replaces the old pin. If the content changed, unpin —
    the old version is stale. The new version starts a fresh fault cycle.
    """
    # Find the latest non-evicted content hash for each pinned path
    latest: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id", "")
            tool_info = tool_use_index.get(tool_use_id, {})
            tool_name = tool_info.get("name", "unknown")
            if tool_name != "Read":
                continue
            tool_input = tool_info.get("input", {})
            key = _eviction_key(tool_name, tool_input)
            if key is None or key not in page_store._pinned:
                continue
            block_content = block.get("content", "")
            # Skip already-evicted summaries
            if isinstance(block_content, str) and (
                block_content.startswith("[tensor:")
                or block_content.startswith("[Paged out:")
            ):
                continue
            latest[key] = _content_hash(block_content)

    # Unpin if the latest read has different content
    for key, ch in latest.items():
        if ch != page_store._pinned[key]:
            page_store.unpin(key)
