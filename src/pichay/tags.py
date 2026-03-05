"""Cleanup tag parser — extracts cooperative memory operations from model text.

The model emits <memory_cleanup> tags inline in its response. The proxy
extracts them on the next request, executes the operations, and strips
the tags from the message text. No SSE stream manipulation needed.

Tag format:
    <memory_cleanup>
    drop: block:a3f2b901
    summarize: block:7e9d4c12 "Model-authored summary of what this block contained"
    anchor: block:c8ad36b2
    release: src/arbiter/evaluator.py, src/arbiter/rules.py
    </memory_cleanup>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CleanupOps:
    """Parsed cleanup operations from a <memory_cleanup> tag."""
    drops: list[str] = field(default_factory=list)
    summaries: list[tuple[str, str]] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    releases: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.drops or self.summaries or self.anchors or self.releases)

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
        return ", ".join(parts) if parts else "no-ops"


# Match <memory_cleanup>...</memory_cleanup> blocks (non-greedy)
_TAG_PATTERN = re.compile(
    r"<memory_cleanup>\s*(.*?)\s*</memory_cleanup>",
    re.DOTALL,
)

# Match block ID references: block:xxxxxxxx (8-12 hex chars, no more)
_BLOCK_ID = re.compile(r"block:([a-f0-9]{8,12})(?![a-f0-9])")

# Match summarize with quoted summary text
_SUMMARIZE_PATTERN = re.compile(
    r'summarize:\s*block:([a-f0-9]{8,12})(?![a-f0-9])\s+"([^"]*)"'
)

# Match release with comma-separated paths
_RELEASE_PATTERN = re.compile(r"release:\s*(.+)")


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

            # Release
            m = _RELEASE_PATTERN.match(line)
            if m:
                paths = [p.strip() for p in m.group(1).split(",") if p.strip()]
                ops.releases.extend(paths)
                continue

    return ops


def strip_cleanup_tags(text: str) -> str:
    """Remove all <memory_cleanup>...</memory_cleanup> tags from text.

    Preserves surrounding text. Cleans up extra blank lines left by
    tag removal.
    """
    result = _TAG_PATTERN.sub("", text)
    # Clean up runs of 3+ newlines left by tag removal
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() if result.strip() != text.strip() else result
