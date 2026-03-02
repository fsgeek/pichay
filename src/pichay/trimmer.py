#!/usr/bin/env python3
"""System prompt trimmer for the proxy.

Three interventions to reduce system prompt waste:

1. Tool definition trimming: Replace unused tool schemas with one-line
   stubs. Re-inject the full definition on first use. The API sends
   ~18 tool definitions per request but only ~5 are used per session —
   45K bytes/request wasted on unused schemas.

2. Skill deduplication: Skills appear tripled under prefixes like
   'pptx', 'example-skills:pptx', 'document-skills:pptx'. Keep only
   the first occurrence of each base skill name.

3. Static component caching: Track content hashes of system prompt
   components across turns. Log which are static and how many bytes
   would be saved (no stripping yet — needs KV cache integration).

Usage:
    # Offline analysis of proxy logs
    python -m pichay.trimmer logs/proxy_*.jsonl

    # With JSON output
    python -m pichay.trimmer --json logs/proxy_*.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pichay.eval import parse_proxy_log


# ---------------------------------------------------------------------------
# Dataclasses — live trimming stats
# ---------------------------------------------------------------------------

@dataclass
class ToolStubStats:
    """Stats from tool definition trimming on one request."""

    total_tools: int = 0
    stubbed_tools: int = 0
    restored_tools: int = 0
    bytes_before: int = 0
    bytes_after: int = 0

    @property
    def bytes_saved(self) -> int:
        return self.bytes_before - self.bytes_after


@dataclass
class SkillDedupeStats:
    """Stats from skill deduplication on one request."""

    total_entries: int = 0
    unique_skills: int = 0
    duplicates_removed: int = 0
    bytes_before: int = 0
    bytes_after: int = 0

    @property
    def bytes_saved(self) -> int:
        return self.bytes_before - self.bytes_after


@dataclass
class StaticCacheStats:
    """Stats from static component hash tracking on one request."""

    total_components: int = 0
    static_components: int = 0
    static_bytes_skippable: int = 0


@dataclass
class TrimResult:
    """Combined result of all trimming interventions on one request."""

    tools: ToolStubStats = field(default_factory=ToolStubStats)
    skills: SkillDedupeStats = field(default_factory=SkillDedupeStats)
    static: StaticCacheStats = field(default_factory=StaticCacheStats)

    @property
    def total_bytes_saved(self) -> int:
        return self.tools.bytes_saved + self.skills.bytes_saved

    @property
    def total_bytes_skippable(self) -> int:
        """Bytes that could be saved with KV cache integration."""
        return self.total_bytes_saved + self.static.static_bytes_skippable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STUB_SCHEMA: dict = {"type": "object", "properties": {}}

# Skill entry pattern (matches analyzer.py _extract_skill_entries)
_SKILL_ENTRY_RE = re.compile(r"^- ([\w:.-]+): (.+?)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# SystemPromptTrimmer
# ---------------------------------------------------------------------------

class SystemPromptTrimmer:
    """Session-scoped trimmer that reduces system prompt waste.

    Instantiated once per proxy session. Tracks state across requests:
    - Which tools have been used (for stub/restore logic)
    - Full tool definitions (for re-injection on first use)
    - Component hashes from previous turn (for static detection)
    """

    def __init__(self, log_fn: Callable[[dict], None] | None = None):
        self.log_fn = log_fn
        # Tool trimming state
        self._used_tools: set[str] = set()
        self._full_tool_defs: dict[str, dict] = {}
        # Static cache state
        self._prev_hashes: dict[str, str] = {}
        # Cumulative stats
        self.cumulative_tools_bytes_saved: int = 0
        self.cumulative_skills_bytes_saved: int = 0
        self.cumulative_requests: int = 0

    def trim(self, body: dict) -> TrimResult:
        """Apply all trimming interventions to a request body.

        Mutates body in place. Returns stats about what was done.
        """
        self.cumulative_requests += 1
        result = TrimResult()

        # 1. Scan messages for tool usage (before trimming tools)
        self._scan_tool_usage(body.get("messages", []))

        # 2. Tool definition trimming
        if "tools" in body and isinstance(body["tools"], list):
            result.tools = self._trim_tools(body)

        # 3. Skill deduplication in message injections
        result.skills = self._dedupe_skills(body.get("messages", []))

        # 4. Static component tracking (log only, no mutation)
        result.static = self._track_static(body)

        # Accumulate
        self.cumulative_tools_bytes_saved += result.tools.bytes_saved
        self.cumulative_skills_bytes_saved += result.skills.bytes_saved

        # Log trimming decisions
        if self.log_fn is not None:
            self.log_fn({
                "type": "trimming",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_num": self.cumulative_requests,
                "tools": asdict(result.tools),
                "skills": asdict(result.skills),
                "static": asdict(result.static),
                "total_bytes_saved": result.total_bytes_saved,
                "cumulative_tools_saved": self.cumulative_tools_bytes_saved,
                "cumulative_skills_saved": self.cumulative_skills_bytes_saved,
            })

        return result

    def summary(self) -> dict:
        """Current state for the health endpoint."""
        return {
            "requests_trimmed": self.cumulative_requests,
            "tools_used": sorted(self._used_tools),
            "tools_known": len(self._full_tool_defs),
            "cumulative_tools_bytes_saved": self.cumulative_tools_bytes_saved,
            "cumulative_skills_bytes_saved": self.cumulative_skills_bytes_saved,
        }

    # --- Internal methods ---

    def _scan_tool_usage(self, messages: list[dict]) -> None:
        """Scan messages for tool_use blocks to track which tools are used."""
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    self._used_tools.add(block.get("name", ""))

    def _trim_tools(self, body: dict) -> ToolStubStats:
        """Replace unused tool definitions with stubs."""
        tools = body["tools"]
        stats = ToolStubStats(total_tools=len(tools))
        stats.bytes_before = len(json.dumps(tools).encode("utf-8"))

        # Store full definitions on first encounter
        for tool in tools:
            name = tool.get("name", "")
            if name and name not in self._full_tool_defs:
                self._full_tool_defs[name] = tool.copy()

        # Build trimmed tool list
        trimmed = []
        for tool in tools:
            name = tool.get("name", "")
            if name in self._used_tools:
                # Restore full definition (ensures we always send the
                # complete schema for tools the model has called)
                full = self._full_tool_defs.get(name, tool)
                trimmed.append(full)
                # Check if this was previously stubbed (restored this turn)
                if tool.get("input_schema") == _STUB_SCHEMA:
                    stats.restored_tools += 1
            else:
                # Stub this tool
                trimmed.append(_make_tool_stub(tool))
                stats.stubbed_tools += 1

        body["tools"] = trimmed
        stats.bytes_after = len(json.dumps(trimmed).encode("utf-8"))
        return stats

    def _dedupe_skills(self, messages: list[dict]) -> SkillDedupeStats:
        """Remove duplicate skill entries from system-reminder blocks."""
        stats = SkillDedupeStats()

        for msg in messages:
            content = msg.get("content", [])

            if isinstance(content, str):
                if "<system-reminder>" not in content:
                    continue
                before = len(content.encode("utf-8"))
                new_text, entries, dupes = _dedupe_skills_text(content)
                if dupes > 0:
                    msg["content"] = new_text
                    after = len(new_text.encode("utf-8"))
                    stats.bytes_before += before
                    stats.bytes_after += after
                    stats.total_entries += entries
                    stats.duplicates_removed += dupes
                    stats.unique_skills += entries - dupes
                continue

            if not isinstance(content, list):
                continue

            for i, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                text = block.get("text", "")
                if not isinstance(text, str):
                    continue
                if "<system-reminder>" not in text:
                    continue

                before = len(text.encode("utf-8"))
                new_text, entries, dupes = _dedupe_skills_text(text)
                if dupes > 0:
                    content[i] = {**block, "text": new_text}
                    after = len(new_text.encode("utf-8"))
                    stats.bytes_before += before
                    stats.bytes_after += after
                    stats.total_entries += entries
                    stats.duplicates_removed += dupes
                    stats.unique_skills += entries - dupes

        return stats

    def _track_static(self, body: dict) -> StaticCacheStats:
        """Track content hashes of system prompt components across turns."""
        stats = StaticCacheStats()
        current_hashes: dict[str, tuple[str, int]] = {}

        # System prompt blocks
        system = body.get("system", [])
        if isinstance(system, list):
            for i, block in enumerate(system):
                text = _block_text(block)
                if not text:
                    continue
                h = _content_hash(text)
                size = len(text.encode("utf-8"))
                current_hashes[f"system_{i}"] = (h, size)
        elif isinstance(system, str):
            h = _content_hash(system)
            size = len(system.encode("utf-8"))
            current_hashes["system_0"] = (h, size)

        # System-reminders in messages
        reminder_idx = 0
        for msg in body.get("messages", []):
            content = msg.get("content", [])
            texts: list[str] = []
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                texts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and isinstance(b.get("text"), str)
                ]
            for text in texts:
                for match in re.finditer(
                    r"<system-reminder>(.*?)</system-reminder>",
                    text,
                    re.DOTALL,
                ):
                    inner = match.group(1).strip()
                    h = _content_hash(inner)
                    size = len(match.group(0).encode("utf-8"))
                    current_hashes[f"reminder_{reminder_idx}"] = (h, size)
                    reminder_idx += 1

        # Compare with previous turn
        stats.total_components = len(current_hashes)
        for key, (h, size) in current_hashes.items():
            prev_h = self._prev_hashes.get(key)
            if prev_h is not None and prev_h == h:
                stats.static_components += 1
                stats.static_bytes_skippable += size

        # Update for next turn
        self._prev_hashes = {k: h for k, (h, _) in current_hashes.items()}

        return stats


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_tool_stub(tool_def: dict) -> dict:
    """Create a minimal stub for an unused tool definition."""
    desc = tool_def.get("description", "")
    # First line, truncated to keep it compact
    first_line = desc.split("\n")[0].strip()
    if first_line.startswith("- "):
        first_line = first_line[2:]
    if len(first_line) > 120:
        first_line = first_line[:117] + "..."
    return {
        "name": tool_def["name"],
        "description": first_line,
        "input_schema": _STUB_SCHEMA,
    }


def _dedupe_skills_text(text: str) -> tuple[str, int, int]:
    """Deduplicate skill entries in system-reminder blocks.

    Returns (new_text, total_entries, duplicates_removed).
    """
    total_entries = 0
    total_dupes = 0

    def _replace_reminder(match: re.Match) -> str:
        nonlocal total_entries, total_dupes
        inner = match.group(1)
        if "skills are available" not in inner:
            return match.group(0)

        lines = inner.split("\n")
        seen_bases: set[str] = set()
        output_lines: list[str] = []
        for line in lines:
            m = _SKILL_ENTRY_RE.match(line)
            if m:
                total_entries += 1
                full_name = m.group(1)
                base = full_name.split(":")[-1] if ":" in full_name else full_name
                if base in seen_bases:
                    total_dupes += 1
                    continue
                seen_bases.add(base)
            output_lines.append(line)

        return "<system-reminder>" + "\n".join(output_lines) + "</system-reminder>"

    new_text = re.sub(
        r"<system-reminder>(.*?)</system-reminder>",
        _replace_reminder,
        text,
        flags=re.DOTALL,
    )
    return new_text, total_entries, total_dupes


def _block_text(block: dict | str) -> str:
    """Extract text content from a system prompt block."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        return block.get("text", "")
    return ""


def _content_hash(text: str) -> str:
    """Short content hash for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Offline analysis — dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrimTurn:
    """Simulated trimming results for one API call."""

    turn: int
    timestamp: str
    # Tool trimming (estimated from usage patterns)
    tools_used_cumulative: int = 0
    tools_stubbable: int = 0
    tool_bytes_saveable: int = 0
    # Skill dedup (exact)
    skill_entries: int = 0
    skill_duplicates: int = 0
    skill_bytes_saved: int = 0
    # Static tracking (exact)
    total_components: int = 0
    static_components: int = 0
    static_bytes_skippable: int = 0


@dataclass
class SessionTrimReport:
    """Aggregate trimming analysis for one proxy log."""

    log_path: str
    total_turns: int = 0
    total_tools_defined: int = 0
    total_tools_used: int = 0
    per_request_tool_bytes: int = 0
    # Aggregates
    total_skill_duplicates: int = 0
    total_skill_bytes_saved: int = 0
    total_static_bytes_skippable: int = 0
    total_tool_bytes_saveable: int = 0
    turns: list[TrimTurn] = field(default_factory=list)

    @property
    def avg_skill_bytes_saved(self) -> float:
        if self.total_turns == 0:
            return 0.0
        return self.total_skill_bytes_saved / self.total_turns

    @property
    def avg_tool_bytes_saveable(self) -> float:
        if self.total_turns == 0:
            return 0.0
        return self.total_tool_bytes_saveable / self.total_turns


# ---------------------------------------------------------------------------
# Offline analysis — engine
# ---------------------------------------------------------------------------

# Approximate bytes for a tool stub entry (name + short desc + minimal schema)
_STUB_BYTES_ESTIMATE = 80


def analyze_trimming(path: Path) -> SessionTrimReport:
    """Simulate trimming on a proxy log and report potential savings.

    Tool definitions are not stored in the log, so tool savings are
    estimated from the inferred per-tool byte overhead and usage counts.
    Skill dedup and static tracking are computed exactly from the logged
    system prompts and messages.
    """
    from pichay.analyzer import (
        _collect_tool_uses,
        _infer_tool_definition_bytes,
        _KNOWN_TOOLS,
    )

    records = parse_proxy_log(path)
    report = SessionTrimReport(log_path=str(path))

    requests = [r for r in records if r.get("type") == "request"]
    if not requests:
        return report

    # Tool usage across the full session
    tool_use_counts = _collect_tool_uses(records)
    tool_def_bytes = _infer_tool_definition_bytes(records)
    report.total_tools_defined = len(_KNOWN_TOOLS)
    report.total_tools_used = len(tool_use_counts)
    report.per_request_tool_bytes = tool_def_bytes

    # Per-tool byte estimate
    per_tool_bytes = (
        tool_def_bytes / len(_KNOWN_TOOLS) if _KNOWN_TOOLS else 0
    )

    # Track state across turns (simulating the live trimmer)
    used_tools: set[str] = set()
    prev_hashes: dict[str, str] = {}

    for turn_idx, req in enumerate(requests):
        messages = req.get("messages_full", [])
        system = req.get("system_prompt_full", [])

        turn = TrimTurn(
            turn=turn_idx + 1,
            timestamp=req.get("timestamp", ""),
        )

        # --- Tool trimming estimate ---
        # Scan messages for tool uses up to this point
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    used_tools.add(block.get("name", ""))

        turn.tools_used_cumulative = len(used_tools)
        stubbable = max(0, len(_KNOWN_TOOLS) - len(used_tools))
        turn.tools_stubbable = stubbable
        turn.tool_bytes_saveable = int(
            stubbable * (per_tool_bytes - _STUB_BYTES_ESTIMATE)
        )

        # --- Skill dedup (exact) ---
        for text in _find_skill_text_blocks(messages):
            _, entries, dupes = _dedupe_skills_text(text)
            if dupes > 0:
                new_text, _, _ = _dedupe_skills_text(text)
                before = len(text.encode("utf-8"))
                after = len(new_text.encode("utf-8"))
                turn.skill_entries += entries
                turn.skill_duplicates += dupes
                turn.skill_bytes_saved += before - after

        # --- Static tracking (exact) ---
        current_hashes: dict[str, tuple[str, int]] = {}

        if isinstance(system, list):
            for i, block in enumerate(system):
                text = _block_text(block)
                if not text:
                    continue
                h = _content_hash(text)
                size = len(text.encode("utf-8"))
                current_hashes[f"system_{i}"] = (h, size)

        reminder_idx = 0
        for msg in messages:
            content = msg.get("content", [])
            texts: list[str] = []
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                texts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and isinstance(b.get("text"), str)
                ]
            for text in texts:
                for match in re.finditer(
                    r"<system-reminder>(.*?)</system-reminder>",
                    text,
                    re.DOTALL,
                ):
                    inner = match.group(1).strip()
                    h = _content_hash(inner)
                    size = len(match.group(0).encode("utf-8"))
                    current_hashes[f"reminder_{reminder_idx}"] = (h, size)
                    reminder_idx += 1

        turn.total_components = len(current_hashes)
        for key, (h, size) in current_hashes.items():
            if prev_hashes.get(key) == h:
                turn.static_components += 1
                turn.static_bytes_skippable += size

        prev_hashes = {k: h for k, (h, _) in current_hashes.items()}

        # Accumulate
        report.turns.append(turn)
        report.total_turns += 1
        report.total_skill_duplicates += turn.skill_duplicates
        report.total_skill_bytes_saved += turn.skill_bytes_saved
        report.total_static_bytes_skippable += turn.static_bytes_skippable
        report.total_tool_bytes_saveable += turn.tool_bytes_saveable

    return report


def _find_skill_text_blocks(messages: list[dict]) -> list[str]:
    """Find text blocks containing skill lists in system-reminders."""
    blocks = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            if "<system-reminder>" in content and "skills are available" in content:
                blocks.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text", "")
            if not isinstance(text, str):
                continue
            if "<system-reminder>" in text and "skills are available" in text:
                blocks.append(text)
    return blocks


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_trim_report(r: SessionTrimReport) -> None:
    """Print human-readable trimming analysis."""
    print(f"\n{'='*65}")
    print("SYSTEM PROMPT TRIMMING ANALYSIS")
    print(f"{'='*65}")
    print(f"Log: {r.log_path}")
    print(f"API calls: {r.total_turns}")

    # Tool trimming
    print(f"\n--- Tool Definition Trimming ---")
    print(f"  Tools defined:     {r.total_tools_defined:>10,}")
    print(f"  Tools used:        {r.total_tools_used:>10,}")
    print(f"  Tool bytes/req:    {r.per_request_tool_bytes:>10,}")
    print(f"  Est. total saved:  {r.total_tool_bytes_saveable:>10,} bytes")
    print(f"  Est. avg saved:    {r.avg_tool_bytes_saveable:>10,.0f} bytes/req")

    # Skill dedup
    print(f"\n--- Skill Deduplication ---")
    print(f"  Total duplicates:  {r.total_skill_duplicates:>10,}")
    print(f"  Total bytes saved: {r.total_skill_bytes_saved:>10,}")
    print(f"  Avg bytes saved:   {r.avg_skill_bytes_saved:>10,.0f} bytes/req")

    # Static caching
    print(f"\n--- Static Component Caching ---")
    print(f"  Total skippable:   {r.total_static_bytes_skippable:>10,} bytes")

    # Per-turn detail
    if r.turns:
        print(f"\nPer-turn:")
        print(
            f"  {'Turn':>4s}  {'Stubs':>5s}  {'ToolSave':>10s}  "
            f"{'SkDupes':>7s}  {'SkSave':>8s}  "
            f"{'Static':>6s}  {'StaticSave':>10s}"
        )
        for t in r.turns:
            print(
                f"  T{t.turn:>3d}  {t.tools_stubbable:>5d}  "
                f"{t.tool_bytes_saveable:>10,}  "
                f"{t.skill_duplicates:>7d}  {t.skill_bytes_saved:>8,}  "
                f"{t.static_components:>6d}  "
                f"{t.static_bytes_skippable:>10,}"
            )

    # Summary
    total_potential = (
        r.total_tool_bytes_saveable
        + r.total_skill_bytes_saved
        + r.total_static_bytes_skippable
    )
    print(f"\n--- Summary ---")
    print(f"  Tool stub savings:     {r.total_tool_bytes_saveable:>12,} bytes")
    print(f"  Skill dedup savings:   {r.total_skill_bytes_saved:>12,} bytes")
    print(f"  Static cache savings:  {r.total_static_bytes_skippable:>12,} bytes")
    print(f"  Total potential:       {total_potential:>12,} bytes")


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0.0%"
    return f"{part / whole:.1%}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze system prompt trimming potential in proxy logs"
    )
    parser.add_argument(
        "logs",
        type=Path,
        nargs="+",
        help="Proxy JSONL log file(s) to analyze",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    reports: list[SessionTrimReport] = []
    for log_path in args.logs:
        if not log_path.exists():
            print(
                f"Warning: {log_path} not found, skipping.",
                file=sys.stderr,
            )
            continue
        reports.append(analyze_trimming(log_path))

    if args.json:
        for r in reports:
            out = asdict(r)
            out["turn_count"] = len(out.pop("turns"))
            out["avg_skill_bytes_saved"] = r.avg_skill_bytes_saved
            out["avg_tool_bytes_saveable"] = r.avg_tool_bytes_saveable
            print(json.dumps(out))
        return

    for r in reports:
        print_trim_report(r)


if __name__ == "__main__":
    main()
