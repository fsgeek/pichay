#!/usr/bin/env python3
"""System prompt waste analyzer for proxy JSONL logs.

Decomposes the system prompt into semantic components, tracks static vs
dynamic content across turns, cross-references tool definitions against
actual usage, and detects duplicates.

The system prompt in Claude Code sessions is a composite of:
    - Agent identity (short, fixed preamble)
    - Conversation instructions (tool usage, tone, git, etc.)
    - Auto memory configuration
    - Environment info (platform, shell, model)
    - Git status snapshot
    - CLAUDE.md project instructions (in system-reminder blocks in messages)
    - Skills list (in system-reminder blocks in messages)
    - Budget reminders (in system-reminder blocks in messages)

Tool JSON schemas are sent via the API's `tools` parameter, which the
proxy captures in `total_request_bytes` but doesn't log separately. We
infer the tool definition overhead from the gap between total_request_bytes
and (system_prompt_bytes + messages_bytes).

Usage:
    python -m pichay.analyzer <proxy-log.jsonl> [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from pichay.eval import parse_proxy_log


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Component:
    """A semantic component of the system prompt or message injection."""

    name: str
    source: str  # "system_prompt" or "message_injection"
    text: str
    bytes: int
    block_index: int = 0  # which block in the system prompt list
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest()[:16]


@dataclass
class ComponentTrack:
    """Tracks a component across turns for static/dynamic analysis."""

    name: str
    hashes: list[str] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)

    @property
    def is_static(self) -> bool:
        return len(set(self.hashes)) <= 1

    @property
    def total_bytes_sent(self) -> int:
        return sum(self.sizes)

    @property
    def unique_bytes(self) -> int:
        """Bytes of the first occurrence — the only send that mattered."""
        return self.sizes[0] if self.sizes else 0

    @property
    def wasted_bytes(self) -> int:
        """Bytes re-sent that were identical to the first send."""
        if not self.is_static or len(self.sizes) <= 1:
            return 0
        return sum(self.sizes[1:])

    @property
    def turns_present(self) -> int:
        return len(self.hashes)

    @property
    def change_count(self) -> int:
        """How many times the content changed between consecutive turns."""
        changes = 0
        for i in range(1, len(self.hashes)):
            if self.hashes[i] != self.hashes[i - 1]:
                changes += 1
        return changes


@dataclass
class DuplicateGroup:
    """A group of components with identical or near-identical content."""

    canonical_name: str
    members: list[str] = field(default_factory=list)
    instance_bytes: int = 0
    total_bytes: int = 0

    @property
    def duplicate_bytes(self) -> int:
        """Bytes wasted by sending copies beyond the first."""
        return max(0, self.total_bytes - self.instance_bytes)


@dataclass
class ToolUsageReport:
    """Cross-reference of defined tools vs actual usage."""

    defined_tools: list[str] = field(default_factory=list)
    used_tools: list[str] = field(default_factory=list)
    unused_tools: list[str] = field(default_factory=list)
    tool_use_counts: dict[str, int] = field(default_factory=dict)
    tool_definition_bytes: int = 0  # inferred from request gap


@dataclass
class SessionAnalysis:
    """Full analysis of system prompt waste for one proxy log."""

    proxy_log: str
    api_calls: int = 0
    # Total bytes
    total_system_prompt_bytes: int = 0
    total_message_injection_bytes: int = 0
    total_tool_definition_bytes: int = 0
    total_request_bytes: int = 0
    # Static vs dynamic
    static_bytes: int = 0
    dynamic_bytes: int = 0
    static_wasted_bytes: int = 0  # re-sent identical content
    # Duplicates (within a single turn)
    duplicate_bytes: int = 0
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    # Tools
    tool_usage: ToolUsageReport = field(default_factory=ToolUsageReport)
    # Per-component tracking
    component_tracks: list[ComponentTrack] = field(default_factory=list)
    # Per-turn decomposition (first turn only, for display)
    sample_decomposition: list[dict] = field(default_factory=list)

    @property
    def total_overhead_bytes(self) -> int:
        """Total system prompt + injection + tool definition bytes."""
        return (
            self.total_system_prompt_bytes
            + self.total_message_injection_bytes
            + self.total_tool_definition_bytes
        )

    @property
    def static_pct(self) -> float:
        total = self.static_bytes + self.dynamic_bytes
        if total == 0:
            return 0.0
        return self.static_bytes / total

    @property
    def unused_tool_bytes(self) -> int:
        """Estimated bytes wasted on unused tool definitions."""
        if not self.tool_usage.defined_tools:
            return 0
        n_defined = len(self.tool_usage.defined_tools)
        n_unused = len(self.tool_usage.unused_tools)
        if n_defined == 0:
            return 0
        per_tool = self.tool_usage.tool_definition_bytes / n_defined
        return int(per_tool * n_unused)


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------

# Section boundaries in the main system prompt text.
_SP_SECTIONS = [
    ("agent_identity", r"^You are a Claude agent"),
    ("conversation_instructions", r"^You are an interactive agent"),
    ("system_section", r"^# System\b"),
    ("doing_tasks", r"^# Doing tasks\b"),
    ("executing_actions", r"^# Executing actions with care\b"),
    ("using_tools", r"^# Using your tools\b"),
    ("tone_and_style", r"^# Tone and style\b"),
    ("auto_memory", r"^# auto memory\b"),
    ("environment", r"^# Environment\b"),
    ("git_status", r"^gitStatus:"),
    ("fast_mode", r"^<fast_mode_info>"),
]

# Patterns for system-reminder injections in messages.
_REMINDER_PATTERNS = [
    ("skills_list", r"skills are available for use"),
    ("budget_reminder", r"USD budget:"),
    ("claude_md", r"# claudeMd\b"),
    ("memory_md", r"# memoryMd\b|MEMORY\.md"),
    ("current_date", r"# currentDate\b"),
    ("todo_reminder", r"TodoWrite tool hasn't been used"),
]


def _extract_sp_components(system_blocks: list[dict]) -> list[Component]:
    """Decompose system prompt blocks into semantic components."""
    components = []

    for block_idx, block in enumerate(system_blocks):
        if not isinstance(block, dict):
            continue
        text = block.get("text", "")
        if not text:
            continue

        # Try to split the text into sections by heading patterns
        sections = _split_into_sections(text)
        if sections:
            for name, section_text in sections:
                components.append(Component(
                    name=name,
                    source="system_prompt",
                    text=section_text,
                    bytes=len(section_text.encode("utf-8")),
                    block_index=block_idx,
                ))
        else:
            # Single undivided block
            components.append(Component(
                name=f"system_block_{block_idx}",
                source="system_prompt",
                text=text,
                bytes=len(text.encode("utf-8")),
                block_index=block_idx,
            ))

    return components


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split system prompt text into named sections by heading patterns."""
    # Find all section starts
    hits: list[tuple[int, str]] = []
    for name, pattern in _SP_SECTIONS:
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            hits.append((m.start(), name))

    if not hits:
        return []

    hits.sort(key=lambda x: x[0])

    # If there's content before the first hit, capture it as preamble
    sections = []
    if hits[0][0] > 0:
        preamble = text[: hits[0][0]].strip()
        if preamble:
            sections.append(("preamble", preamble))

    for i, (start, name) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((name, section_text))

    return sections


def _extract_message_injections(
    messages: list[dict],
) -> list[Component]:
    """Extract system-reminder injections from message content blocks."""
    components = []

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            if isinstance(content, str) and "<system-reminder>" in content:
                for comp in _parse_reminder_text(content):
                    components.append(comp)
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text", block.get("content", ""))
            if not isinstance(text, str):
                continue
            if "<system-reminder>" not in text:
                continue
            for comp in _parse_reminder_text(text):
                components.append(comp)

    return components


def _parse_reminder_text(text: str) -> list[Component]:
    """Parse system-reminder tags out of a text block."""
    components = []
    for match in re.finditer(
        r"<system-reminder>(.*?)</system-reminder>", text, re.DOTALL
    ):
        inner = match.group(1).strip()
        if not inner:
            continue
        name = _classify_reminder(inner)
        components.append(Component(
            name=name,
            source="message_injection",
            text=inner,
            bytes=len(match.group(0).encode("utf-8")),
        ))
    return components


def _classify_reminder(text: str) -> str:
    """Classify a system-reminder block by its content."""
    for name, pattern in _REMINDER_PATTERNS:
        if re.search(pattern, text):
            return name
    return "unknown_reminder"


# ---------------------------------------------------------------------------
# Skill duplicate detection
# ---------------------------------------------------------------------------

def _extract_skill_entries(text: str) -> list[tuple[str, str]]:
    """Parse skill entries from a skills list block.

    Returns (full_name, base_name) pairs.
    E.g. ("example-skills:pptx", "pptx"), ("pptx", "pptx")
    """
    entries = []
    for match in re.finditer(
        r"^- ([\w:.-]+): (.+?)$", text, re.MULTILINE
    ):
        full_name = match.group(1)
        base = full_name.split(":")[-1] if ":" in full_name else full_name
        entries.append((full_name, base))
    return entries


def _find_duplicate_skills(components: list[Component]) -> list[DuplicateGroup]:
    """Find skills that appear multiple times under different prefixes."""
    groups: list[DuplicateGroup] = []

    for comp in components:
        if comp.name != "skills_list":
            continue

        entries = _extract_skill_entries(comp.text)
        if not entries:
            continue

        # Group by base name
        by_base: dict[str, list[str]] = {}
        for full_name, base in entries:
            by_base.setdefault(base, []).append(full_name)

        for base, names in by_base.items():
            if len(names) <= 1:
                continue
            # Estimate bytes per skill entry (average across the block)
            avg_bytes = comp.bytes // max(1, len(entries))
            groups.append(DuplicateGroup(
                canonical_name=base,
                members=names,
                instance_bytes=avg_bytes,
                total_bytes=avg_bytes * len(names),
            ))

    return groups


# ---------------------------------------------------------------------------
# Tool usage tracking
# ---------------------------------------------------------------------------

# Known Claude Code built-in tools.
_KNOWN_TOOLS = [
    "Agent", "Bash", "Edit", "Glob", "Grep", "Read", "Write",
    "WebFetch", "WebSearch", "NotebookEdit", "TodoWrite", "Skill",
    "AskUserQuestion", "EnterPlanMode", "ExitPlanMode", "TaskOutput",
    "TaskStop", "EnterWorktree",
]


def _collect_tool_uses(records: list[dict]) -> dict[str, int]:
    """Count tool_use occurrences across all request messages."""
    counts: dict[str, int] = {}
    for rec in records:
        if rec.get("type") != "request":
            continue
        for msg in rec.get("messages_full", []):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    counts[name] = counts.get(name, 0) + 1
    return counts


def _infer_tool_definition_bytes(records: list[dict]) -> int:
    """Infer tool definition overhead from the request size gap.

    total_request_bytes = system_prompt + messages + tools + metadata.
    The gap between total and (system + messages) is mostly tool schemas.
    """
    gaps = []
    for rec in records:
        if rec.get("type") != "request":
            continue
        total = rec.get("total_request_bytes", 0)
        sys_bytes = rec.get("system", {}).get("system_prompt_bytes", 0)
        msg_bytes = rec.get("messages", {}).get("messages_total_bytes", 0)
        gap = total - sys_bytes - msg_bytes
        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return 0

    # The gap should be roughly constant (tool schemas don't change).
    # Use the median to be robust against outliers.
    gaps.sort()
    return gaps[len(gaps) // 2]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_system_prompts(proxy_path: Path) -> SessionAnalysis:
    """Analyze system prompt waste across a proxy JSONL log."""
    records = parse_proxy_log(proxy_path)
    analysis = SessionAnalysis(proxy_log=str(proxy_path))

    # Separate request records
    requests = [r for r in records if r.get("type") == "request"]
    analysis.api_calls = len(requests)
    if not requests:
        return analysis

    # Tool usage
    tool_use_counts = _collect_tool_uses(records)
    tool_def_bytes = _infer_tool_definition_bytes(records)
    analysis.tool_usage = ToolUsageReport(
        defined_tools=list(_KNOWN_TOOLS),
        used_tools=sorted(tool_use_counts.keys()),
        unused_tools=sorted(
            set(_KNOWN_TOOLS) - set(tool_use_counts.keys())
        ),
        tool_use_counts=tool_use_counts,
        tool_definition_bytes=tool_def_bytes,
    )
    analysis.total_tool_definition_bytes = tool_def_bytes * len(requests)

    # Component tracking across turns
    tracks: dict[str, ComponentTrack] = {}

    for turn_idx, rec in enumerate(requests):
        sp = rec.get("system_prompt_full", [])
        msgs = rec.get("messages_full", [])

        # Extract components
        sp_components = _extract_sp_components(
            sp if isinstance(sp, list) else []
        )
        msg_components = _extract_message_injections(msgs)

        # System prompt bytes
        sp_bytes = sum(c.bytes for c in sp_components)
        analysis.total_system_prompt_bytes += sp_bytes

        # Message injection bytes
        inj_bytes = sum(c.bytes for c in msg_components)
        analysis.total_message_injection_bytes += inj_bytes

        analysis.total_request_bytes += rec.get("total_request_bytes", 0)

        # Track each component
        all_components = sp_components + msg_components
        for comp in all_components:
            if comp.name not in tracks:
                tracks[comp.name] = ComponentTrack(name=comp.name)
            tracks[comp.name].hashes.append(comp.content_hash)
            tracks[comp.name].sizes.append(comp.bytes)

        # Duplicate detection (skills within this turn)
        if turn_idx == 0:
            dupe_groups = _find_duplicate_skills(msg_components)
            analysis.duplicate_groups = dupe_groups
            analysis.duplicate_bytes = sum(
                g.duplicate_bytes for g in dupe_groups
            )

        # Sample decomposition (first turn)
        if turn_idx == 0:
            analysis.sample_decomposition = [
                {"name": c.name, "source": c.source, "bytes": c.bytes}
                for c in all_components
            ]

    # Aggregate static/dynamic
    for track in tracks.values():
        if track.is_static:
            analysis.static_bytes += track.total_bytes_sent
            analysis.static_wasted_bytes += track.wasted_bytes
        else:
            analysis.dynamic_bytes += track.total_bytes_sent

    analysis.component_tracks = sorted(
        tracks.values(), key=lambda t: t.total_bytes_sent, reverse=True
    )

    return analysis


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_analysis(a: SessionAnalysis) -> None:
    """Print human-readable analysis."""
    print(f"\n{'=' * 65}")
    print("SYSTEM PROMPT WASTE ANALYSIS")
    print(f"{'=' * 65}")
    print(f"Log: {a.proxy_log}")
    print(f"API calls: {a.api_calls}")

    # Component decomposition (first turn)
    print(f"\n--- Component Decomposition (Turn 1) ---")
    for comp in a.sample_decomposition:
        print(f"  {comp['name']:<35s} {comp['bytes']:>8,} bytes  ({comp['source']})")

    # Static vs dynamic
    total_content = a.static_bytes + a.dynamic_bytes
    print(f"\n--- Static vs Dynamic ---")
    print(f"  Static content:       {a.static_bytes:>12,} bytes  ({_pct(a.static_bytes, total_content)})")
    print(f"  Dynamic content:      {a.dynamic_bytes:>12,} bytes  ({_pct(a.dynamic_bytes, total_content)})")
    print(f"  Static re-send waste: {a.static_wasted_bytes:>12,} bytes")

    # Per-component tracking
    print(f"\n--- Per-Component Tracking ---")
    print(f"  {'Component':<35s} {'Total':>10s} {'Turns':>6s} {'Changes':>8s} {'Static':>7s}")
    print(f"  {'-'*35} {'-'*10} {'-'*6} {'-'*8} {'-'*7}")
    for t in a.component_tracks:
        static_label = "yes" if t.is_static else "no"
        print(
            f"  {t.name:<35s} {t.total_bytes_sent:>10,} "
            f"{t.turns_present:>6d} {t.change_count:>8d} {static_label:>7s}"
        )

    # Tool usage
    tu = a.tool_usage
    print(f"\n--- Tool Usage ---")
    print(f"  Tool definition overhead: {tu.tool_definition_bytes:>10,} bytes/request")
    print(f"  Total across session:     {a.total_tool_definition_bytes:>10,} bytes")
    print(f"  Defined tools:  {len(tu.defined_tools)}")
    print(f"  Used tools:     {len(tu.used_tools)}  {tu.used_tools}")
    print(f"  Unused tools:   {len(tu.unused_tools)}  {tu.unused_tools}")
    if tu.defined_tools:
        print(f"  Est. unused tool bytes:   {a.unused_tool_bytes:>10,} bytes/request")
    if tu.tool_use_counts:
        print(f"\n  Tool call counts:")
        for name, count in sorted(
            tu.tool_use_counts.items(), key=lambda x: -x[1]
        ):
            print(f"    {name:<25s} {count:>6d}")

    # Duplicates
    if a.duplicate_groups:
        print(f"\n--- Duplicate Content ---")
        print(f"  Total duplicate bytes (per turn): {a.duplicate_bytes:>10,}")
        for g in sorted(a.duplicate_groups, key=lambda x: -x.duplicate_bytes):
            print(
                f"  {g.canonical_name:<30s} "
                f"x{len(g.members)} copies, "
                f"{g.duplicate_bytes:>6,} bytes wasted"
            )
            for m in g.members:
                print(f"    - {m}")

    # Summary
    print(f"\n--- Session Summary ---")
    print(f"  Total request bytes:          {a.total_request_bytes:>12,}")
    print(f"  System prompt bytes:          {a.total_system_prompt_bytes:>12,}  ({_pct(a.total_system_prompt_bytes, a.total_request_bytes)})")
    print(f"  Message injection bytes:      {a.total_message_injection_bytes:>12,}  ({_pct(a.total_message_injection_bytes, a.total_request_bytes)})")
    print(f"  Tool definition bytes:        {a.total_tool_definition_bytes:>12,}  ({_pct(a.total_tool_definition_bytes, a.total_request_bytes)})")
    print(f"  Total overhead:               {a.total_overhead_bytes:>12,}  ({_pct(a.total_overhead_bytes, a.total_request_bytes)})")
    print(f"  Static re-send waste:         {a.static_wasted_bytes:>12,}")
    print(f"  Duplicate waste (per turn):   {a.duplicate_bytes:>12,}")
    print(f"  Unused tool waste (per req):  {a.unused_tool_bytes:>12,}")


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0.0%"
    return f"{part / whole:.1%}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze system prompt waste in proxy JSONL logs"
    )
    parser.add_argument(
        "proxy_log",
        type=Path,
        help="Path to proxy JSONL log file",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    if not args.proxy_log.exists():
        print(f"Error: {args.proxy_log} not found", file=sys.stderr)
        sys.exit(1)

    analysis = analyze_system_prompts(args.proxy_log)

    if args.json:
        out = asdict(analysis)
        # Trim large fields
        out.pop("sample_decomposition", None)
        ct = out.pop("component_tracks", None)
        if ct:
            out["component_tracks"] = [
                {
                    "name": t["name"],
                    "total_bytes_sent": sum(t["sizes"]),
                    "turns_present": len(t["hashes"]),
                    "is_static": len(set(t["hashes"])) <= 1,
                    "change_count": sum(
                        1 for i in range(1, len(t["hashes"]))
                        if t["hashes"][i] != t["hashes"][i - 1]
                    ),
                }
                for t in ct
            ]
        print(json.dumps(out, indent=2))
    else:
        print_analysis(analysis)


if __name__ == "__main__":
    main()
