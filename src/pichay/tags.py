"""Cleanup tag parser — extracts cooperative memory operations from model text.

The model emits <memory_cleanup> tags inline in its response. The proxy
extracts them on the next request, executes the operations, and strips
the tags from the message text. No SSE stream manipulation needed.

Tag format:
    <memory_cleanup>
    drop: tensor:a3f2b901
    summarize: tensor:7e9d4c12 "Model-authored summary of what this block contained"
    anchor: tensor:c8ad36b2
    release: src/arbiter/evaluator.py, src/arbiter/rules.py
    </memory_cleanup>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CollapseOp:
    """A turn-range collapse: replace multiple turns with a summary."""
    start_turn: int
    end_turn: int
    summary: str


@dataclass
class CleanupOps:
    """Parsed cleanup operations from a <memory_cleanup> tag."""
    drops: list[str] = field(default_factory=list)
    summaries: list[tuple[str, str]] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    releases: list[str] = field(default_factory=list)
    collapses: list[CollapseOp] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.drops or self.summaries or self.anchors
                    or self.releases or self.collapses)

    def __str__(self) -> str:
        parts = []
        if self.drops:
            parts.append(f"drop={len(self.drops)}")
        if self.summaries:
            parts.append(f"summarize={len(self.summaries)}")
        if self.anchors:
            parts.append(f"anchor={len(self.anchors)}")
        if self.releases:
            parts.append(f"release={len(self.releases)}")
        if self.collapses:
            parts.append(f"collapse={len(self.collapses)}")
        return ", ".join(parts) if parts else "no-ops"


# Match <memory_cleanup>...</memory_cleanup> blocks (non-greedy)
_TAG_PATTERN = re.compile(
    r"<memory_cleanup>\s*(.*?)\s*</memory_cleanup>",
    re.DOTALL,
)

# Match tensor/block ID references: tensor:xxxxxxxx or block:xxxxxxxx (8-12 hex chars)
_BLOCK_ID = re.compile(r"(?:tensor|block):([a-f0-9]{8,12})(?![a-f0-9])")

# Match summarize with quoted summary text
_SUMMARIZE_PATTERN = re.compile(
    r'summarize:\s*(?:tensor|block):([a-f0-9]{8,12})(?![a-f0-9])\s+"([^"]*)"'
)

# Match release with comma-separated paths
_RELEASE_PATTERN = re.compile(r"release:\s*(.+)")

# Match collapse with turn range and quoted summary
# Format: collapse: turns 3-8 "Summary of what happened in those turns"
_COLLAPSE_PATTERN = re.compile(
    r'collapse:\s*turns\s+(\d+)\s*-\s*(\d+)\s+"([^"]*)"'
)


def parse_cleanup_tags(text: str) -> CleanupOps:
    """Extract cleanup operations from text containing <memory_cleanup> tags.

    Returns a CleanupOps with all parsed operations. Multiple tags in
    the same text are merged into a single CleanupOps.
    """
    ops = CleanupOps()

    for match in _TAG_PATTERN.finditer(text):
        body = match.group(1)
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue

            # Summarize (must check before drop — both start with block ID)
            m = _SUMMARIZE_PATTERN.match(line)
            if m:
                ops.summaries.append((m.group(1), m.group(2)))
                continue

            # Drop
            if line.startswith("drop:"):
                m = _BLOCK_ID.search(line)
                if m:
                    ops.drops.append(m.group(1))
                continue

            # Anchor
            if line.startswith("anchor:"):
                m = _BLOCK_ID.search(line)
                if m:
                    ops.anchors.append(m.group(1))
                continue

            # Collapse (turn range)
            m = _COLLAPSE_PATTERN.match(line)
            if m:
                ops.collapses.append(CollapseOp(
                    start_turn=int(m.group(1)),
                    end_turn=int(m.group(2)),
                    summary=m.group(3),
                ))
                continue

            # Release
            m = _RELEASE_PATTERN.match(line)
            if m:
                paths = [p.strip() for p in m.group(1).split(",") if p.strip()]
                ops.releases.extend(paths)
                continue

    return ops


# Match <yuyay-response>...</yuyay-response> blocks
_YUYAY_RESPONSE_PATTERN = re.compile(
    r"<yuyay-response>\s*(.*?)\s*</yuyay-response>",
    re.DOTALL,
)

# Match structured eviction decisions: <release handle="abc123"/>
_YUYAY_RELEASE = re.compile(r'<release\s+handle="([a-f0-9]{8,12})"')
# Match structured retain (logged but no action needed)
_YUYAY_RETAIN = re.compile(r'<retain\s+handle="([a-f0-9]{8,12})"')


def parse_yuyay_response(text: str) -> CleanupOps:
    """Extract memory operations from <yuyay-response> blocks.

    The model responds to <yuyay-query> with structured eviction
    decisions. These are converted to CleanupOps for execution
    through the same pipeline as <memory_cleanup> tags.

    Supports two formats:
    - Structured XML: <release handle="abc123"/>
    - Prose with release directives: release: tensor:abc123
    """
    ops = CleanupOps()

    for match in _YUYAY_RESPONSE_PATTERN.finditer(text):
        body = match.group(1)

        # Try structured XML format first
        for m in _YUYAY_RELEASE.finditer(body):
            ops.releases.append(m.group(1))

        # Also try the prose release format (same as cleanup tags)
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _RELEASE_PATTERN.match(line)
            if m:
                paths = [p.strip() for p in m.group(1).split(",") if p.strip()]
                ops.releases.extend(paths)

    return ops


def strip_yuyay_tags(text: str) -> str:
    """Remove <yuyay-response> blocks from model output."""
    result = _YUYAY_RESPONSE_PATTERN.sub("", text)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() if result.strip() != text.strip() else result


def strip_cleanup_tags(text: str) -> str:
    """Remove all <memory_cleanup>...</memory_cleanup> tags from text.

    Preserves surrounding text. Cleans up extra blank lines left by
    tag removal.
    """
    result = _TAG_PATTERN.sub("", text)
    # Clean up runs of 3+ newlines left by tag removal
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() if result.strip() != text.strip() else result
