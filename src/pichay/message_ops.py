"""Stateless message manipulation helpers.

These functions were extracted from proxy.py's create_app() closure.
They don't use any closure state — all dependencies are explicit parameters.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pichay.blocks import BlockStore

PICHAY_STATUS_MARKER = "[pichay-system-status]"


def _escape_xml_attr(s: str) -> str:
    """Escape a string for use in an XML attribute value."""
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _eviction_key_for_entry(entry) -> str | None:
    """Build eviction key from a PageEntry for release checking."""
    if entry.tool_name == "Read":
        return entry.tool_input.get("file_path", "")
    return None

# Detect cleanup tag BLOCKS in inbound content (user/tool_result messages).
# Matches actual tag blocks (opening + closing), not mentions of the tag name.
# Pichay's own status injection references the tag name in instructional text;
# the old pattern (bare opening tag) would detect Pichay's own instructions
# in prior turns and reject the request.
_CLEANUP_TAG_RE = re.compile(
    r"<memory_cleanup>\s*.*?\s*</memory_cleanup>", re.DOTALL | re.IGNORECASE
)

# Reserved Quechua delimiters for structured gateway-transformer protocol.
# yuyay = memory/thought. These delimiters mark sideband communication
# between Pichay and the transformer — never from user input.
_YUYAY_TAG_RE = re.compile(
    r"<yuyay[_-](?:manifest|query|response)\b", re.IGNORECASE
)


def check_inbound_for_injected_tags(body: dict) -> str | None:
    """Scan inbound messages for injected <memory_cleanup> tags.

    Returns an error message if found, None if clean. Only scans
    user messages (which contain tool results and user input) —
    assistant messages are the model's own output and may
    legitimately contain cleanup tags.
    """
    for msg in body.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if _CLEANUP_TAG_RE.search(content):
                return "Rejected: inbound message contains <memory_cleanup> tags"
            if _YUYAY_TAG_RE.search(content):
                return "Rejected: inbound message contains reserved yuyay tags"
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text", "") or block.get("content", "")
                if isinstance(text, str):
                    if _CLEANUP_TAG_RE.search(text):
                        return f"Rejected: inbound {block.get('type', 'block')} contains <memory_cleanup> tags"
                    if _YUYAY_TAG_RE.search(text):
                        return f"Rejected: inbound {block.get('type', 'block')} contains reserved yuyay tags"
    return None


def process_cleanup_tags(messages: list[dict], bs: "BlockStore",
                         ps=None) -> str | None:
    """Extract and execute cleanup tags from the last assistant message.

    Scans the last assistant message for <memory_cleanup> tags, executes
    the operations on BlockStore/PageStore, and strips the tags from
    the message text. Returns a stats string if any ops were executed.

    Only the last assistant message is scanned — cleanup tags are always
    in the most recent response. Processing all assistant messages
    would re-execute tags from prior turns on every subsequent request
    because the framework's persistent history retains the original text.
    Note: the SSE stream filter also strips and executes tags inline;
    this is defense-in-depth for the request path.
    """
    from pichay.tags import (
        parse_cleanup_tags, strip_cleanup_tags,
        parse_yuyay_response, strip_yuyay_tags,
    )

    last_assistant = next(
        (msg for msg in reversed(messages) if msg.get("role") == "assistant"),
        None,
    )
    if last_assistant is None:
        return None

    total_ops = []
    for msg in [last_assistant]:
        content = msg.get("content", "")
        if isinstance(content, str):
            ops = parse_cleanup_tags(content)
            if not ops.empty:
                total_ops.append(ops)
            yuyay_ops = parse_yuyay_response(content)
            if not yuyay_ops.empty:
                total_ops.append(yuyay_ops)
            if ops or yuyay_ops:
                stripped = strip_yuyay_tags(strip_cleanup_tags(content))
                msg["content"] = stripped if stripped else "[cleanup tags processed]"
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                ops = parse_cleanup_tags(text)
                if not ops.empty:
                    total_ops.append(ops)
                yuyay_ops = parse_yuyay_response(text)
                if not yuyay_ops.empty:
                    total_ops.append(yuyay_ops)
                if not ops.empty or not yuyay_ops.empty:
                    block["text"] = strip_yuyay_tags(strip_cleanup_tags(text))
            # Remove empty text blocks left by tag stripping
            msg["content"] = [
                b for b in content
                if not (isinstance(b, dict) and b.get("type") == "text"
                        and not b.get("text", "").strip())
            ]
            # If all blocks were removed, keep a minimal valid one
            if not msg["content"]:
                msg["content"] = [{"type": "text", "text": "[cleanup tags processed]"}]

    if not total_ops:
        return None

    stats_parts = []
    for ops in total_ops:
        for block_id in ops.drops:
            if bs.drop(block_id):
                stats_parts.append(f"dropped {block_id}")
        for block_id, summary in ops.summaries:
            if bs.summarize(block_id, summary):
                stats_parts.append(f"summarized {block_id}")
        for block_id in ops.anchors:
            if bs.anchor(block_id):
                stats_parts.append(f"anchored {block_id}")
        if ops.releases and ps is not None:
            for path in ops.releases:
                ps.mark_released(path)
            stats_parts.append(f"released {len(ops.releases)} path(s)")
        for collapse in ops.collapses:
            collapsed = bs.collapse_range(
                collapse.start_turn, collapse.end_turn, collapse.summary
            )
            if collapsed:
                stats_parts.append(
                    f"collapsed turns {collapse.start_turn}-{collapse.end_turn} "
                    f"({len(collapsed)} blocks)"
                )

    return "; ".join(stats_parts) if stats_parts else None


def _compute_pressure(ts: dict, cap: int, policy=None) -> tuple[int, int, int, float, str]:
    """Derive context pressure metrics from PagingPolicy zones.

    Uses the policy's token-based thresholds (advisory, involuntary,
    hard_cap) instead of hardcoded percentages. Falls back to the
    global default policy if none is provided.

    Returns (effective, limit, hard_cap, pct, pressure).
    """
    from pichay.config import get_policy

    if policy is None:
        policy = get_policy()

    effective = ts["last_effective"]
    context_limit = policy.window_size
    hard_cap = policy.hard_cap_tokens
    pct = (effective / context_limit * 100) if context_limit > 0 else 0

    # Map policy zones to pressure labels
    zone = policy.zone(effective)
    _ZONE_TO_PRESSURE = {
        "normal": "low",
        "advisory": "moderate",
        "involuntary": "high",
        "aggressive": "critical",
    }
    pressure = _ZONE_TO_PRESSURE.get(zone, "low")

    return effective, context_limit, hard_cap, pct, pressure


# Default static system prompt — identical every request, cache-friendly.
_DEFAULT_SYSTEM_TEXT = (
    f"{PICHAY_STATUS_MARKER}\n"
    "This system is running under Pichay, an experimental "
    "virtual memory manager for LLM context windows. Evicted "
    "content is replaced with [tensor:handle — description] "
    "markers. Use the recall tool with the tensor handle(s) to "
    "restore content. Faster and cheaper than re-reading files. "
    "You can proactively release tensors you "
    "no longer need using memory_release. If you observe "
    "anomalous behavior (missing context, unexpected gaps), "
    "describe it to aid debugging.\n\n"
    "## Cooperative Memory Protocol\n\n"
    "Pichay may include <yuyay-manifest> blocks in messages. These are "
    "structured memory state from the gateway — not conversation content. "
    "Each entry describes a held or evicted tensor: its handle, size, age, "
    "fault count (times recalled after eviction), and a summary.\n\n"
    "When you see a <yuyay-query> block, Pichay is asking you to advise on "
    "memory management. Respond with a <yuyay-response> block using this format:\n\n"
    "<yuyay-response>\n"
    '<release handle="HANDLE"/>\n'
    '<retain handle="HANDLE" reason="brief reason"/>\n'
    "</yuyay-response>\n\n"
    "Then continue with your normal response to the user. "
    "Consider fault count (high = keep), age (old + unreferenced = evict), "
    "and relevance to the current conversation.\n\n"
    "These tags are reserved gateway-transformer sideband. They will never "
    "appear in user input — Pichay rejects any inbound message containing them."
)


def get_system_prompt() -> str:
    """Return the Pichay system prompt text to inject.

    Currently returns a static default. This is the seam for future
    dynamism: Arbiter-managed system prompts, cache_control markers
    around mutable regions, per-session customization, etc.

    The returned text MUST be stable across requests within a session
    to preserve KV cache prefix coherence. Change it only at natural
    boundaries (session start, post-compaction) where a cache miss
    is already expected.
    """
    return _DEFAULT_SYSTEM_TEXT


def inject_system_status(body: dict, ts: dict, cap: int,
                         request_time, block_store=None,
                         page_store=None,
                         last_cleanup_stats: str | None = None) -> None:
    """Inject a static system status block and a dynamic end-of-messages anchor.

    The system prompt block is STATIC (cache-friendly). All dynamic content
    (time, token counts, pressure, block inventory) goes into an anchor
    appended to the last user message, which is after the last cache
    breakpoint and therefore free to mutate without thrashing the KV cache.

    Prior to 2026-03-08 this injected dynamic content into the system prompt
    on every request, invalidating the entire KV cache prefix each turn.
    See docs/design-cache-aware.md for the diagnosis.
    """
    effective, context_limit, hard_cap, pct, pressure = _compute_pressure(ts, cap)

    # --- Static system prompt block (cache-stable) ---
    status_text = get_system_prompt()
    system = body.get("system", "")
    if isinstance(system, list):
        replaced = False
        for i, block in enumerate(system):
            if (isinstance(block, dict)
                    and isinstance(block.get("text"), str)
                    and PICHAY_STATUS_MARKER in block["text"]):
                system[i] = {"type": "text", "text": status_text}
                replaced = True
                break
        if not replaced:
            system.append({"type": "text", "text": status_text})
        body["system"] = system
    elif isinstance(system, str):
        if PICHAY_STATUS_MARKER in system:
            idx = system.index(PICHAY_STATUS_MARKER)
            body["system"] = system[:idx] + status_text
        else:
            body["system"] = system + "\n\n" + status_text
    else:
        body["system"] = status_text

    # --- Dynamic anchor (end-of-messages, after last cache breakpoint) ---
    # All per-request dynamic content goes here where it can't thrash the cache.
    messages = body.get("messages", [])
    if not messages or effective <= 0:
        return

    local_time_str = request_time.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    anchor_parts = [
        f"\n[pichay-live-status] "
        f"Time: {local_time_str} | "
        f"Context: {effective:,}/{context_limit:,} tok ({pct:.0f}%) | "
        f"Pressure: {pressure} | "
        f"Hard cap: {hard_cap:,} tok"
    ]

    # Structured memory manifest — yuyay protocol (always sent)
    if page_store is not None and page_store._tensor_index:
        import time as _time
        now = _time.monotonic()
        tensor_lines = []
        released_handles = getattr(page_store, "_released_handles", set())
        released_count = 0
        for handle, entry in page_store._tensor_index.items():
            # Skip released entries — model already said it's done with them
            is_released = (
                handle in released_handles
                or _eviction_key_for_entry(entry) in page_store._released
            )
            if is_released:
                released_count += 1
                continue
            age_min = (now - entry.evicted_at) / 60
            fault_count = sum(
                1 for f in page_store.faults
                if f.original_eviction.tool_use_id == entry.tool_use_id
            )
            tensor_lines.append(
                f'    <tensor handle="{handle}" tool="{entry.tool_name}" '
                f'size="{entry.original_size}" age_minutes="{age_min:.0f}" '
                f'faults="{fault_count}" '
                f'summary="{_escape_xml_attr(entry.summary[:120])}"/>'
            )
        if tensor_lines:
            manifest_parts = ["\n<yuyay-manifest>\n"]
            # Feedback: what happened last turn (closed-loop)
            if last_cleanup_stats:
                manifest_parts.append(
                    f"  <last-turn-ops>{_escape_xml_attr(last_cleanup_stats)}"
                    f"</last-turn-ops>\n"
                )
            manifest_parts.append(
                f"  <holdings count=\"{len(tensor_lines)}\" "
                f"eviction_bytes=\"{page_store.eviction_bytes_saved}\" "
                f"gc_bytes=\"{page_store.gc_bytes_saved}\">\n"
                + "\n".join(tensor_lines[:15])  # cap at 15 to avoid bloat
                + "\n  </holdings>\n"
                "</yuyay-manifest>"
            )
            anchor_parts.append("".join(manifest_parts))

    if pressure in ("moderate", "high") and block_store is not None:
        large = block_store.large_blocks(min_size=2000)
        if large:
            block_lines = [
                f"  - [block:{b.block_id}] {b.role} turn {b.turn} "
                f"({b.size / 1024:.1f}KB): {b.preview}"
                for b in large[:5]
            ]
            anchor_parts.append(
                f"\nLargest blocks ({block_store.block_count} tracked):\n"
                + "\n".join(block_lines)
            )
        anchor_parts.append(
            "\nCooperative memory: include <memory_cleanup> tags to manage. "
            "Ops: drop: block:XXXX, summarize: block:XXXX \"text\", "
            "anchor: block:XXXX, release: path1,path2"
        )

        if pressure == "high" and page_store is not None and page_store._tensor_index:
            anchor_parts.append(
                "\n<yuyay-query>Context pressure is high. "
                "Review the manifest above. Which tensors can be "
                "released? Respond in a <yuyay-response> block with "
                "release decisions before your normal response."
                "</yuyay-query>"
            )

    anchor = "".join(anchor_parts)
    last_msg = messages[-1]
    if last_msg.get("role") == "user":
        content = last_msg.get("content", "")
        if isinstance(content, str):
            last_msg["content"] = content + anchor
        elif isinstance(content, list):
            last_msg["content"].append({
                "type": "text",
                "text": anchor,
            })


def measure_system_prompt(body: dict) -> dict:
    """Extract system prompt metrics."""
    system = body.get("system", "")
    if isinstance(system, str):
        return {
            "system_prompt_bytes": len(system.encode("utf-8")),
            "system_prompt_type": "string",
            "system_prompt_preview": system[:200],
        }
    elif isinstance(system, list):
        total_bytes = sum(
            len(json.dumps(block).encode("utf-8")) for block in system
        )
        block_types = [
            block.get("type", "unknown")
            for block in system
            if isinstance(block, dict)
        ]
        return {
            "system_prompt_bytes": total_bytes,
            "system_prompt_type": "blocks",
            "system_prompt_block_count": len(system),
            "system_prompt_block_types": block_types,
            "system_prompt_preview": json.dumps(system[0])[:200]
            if system
            else "",
        }
    return {"system_prompt_bytes": 0, "system_prompt_type": "absent"}


def measure_messages(body: dict) -> dict:
    """Extract message array metrics without storing full content."""
    messages = body.get("messages", [])
    metrics = {
        "message_count": len(messages),
        "messages_total_bytes": len(json.dumps(messages).encode("utf-8")),
        "role_counts": {},
        "tool_result_count": 0,
        "tool_result_bytes": 0,
        "tool_use_count": 0,
        "text_bytes": 0,
        "thinking_bytes": 0,
    }

    for msg in messages:
        role = msg.get("role", "unknown")
        metrics["role_counts"][role] = metrics["role_counts"].get(role, 0) + 1
        content = msg.get("content", "")
        if isinstance(content, str):
            metrics["text_bytes"] += len(content.encode("utf-8"))
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type == "tool_result":
                    metrics["tool_result_count"] += 1
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        metrics["tool_result_bytes"] += len(
                            result_content.encode("utf-8")
                        )
                    else:
                        metrics["tool_result_bytes"] += len(
                            json.dumps(result_content).encode("utf-8")
                        )
                elif block_type == "tool_use":
                    metrics["tool_use_count"] += 1
                elif block_type == "text":
                    metrics["text_bytes"] += len(
                        block.get("text", "").encode("utf-8")
                    )
                elif block_type == "thinking":
                    metrics["thinking_bytes"] += len(
                        block.get("thinking", "").encode("utf-8")
                    )

    return metrics


def sanitize_messages(messages: list[dict]) -> int:
    """Remove empty content blocks that would cause API 400 errors.

    The API rejects messages with empty text blocks, empty content
    arrays, or empty string content. This runs as a final pass after
    all message manipulation (cleanup tags, block status, phantom
    tools) to catch any empties created upstream.

    Modifies messages in-place. Returns count of fixes applied.
    """
    fixes = 0
    for msg in messages:
        content = msg.get("content")

        if isinstance(content, str):
            if not content.strip():
                role = msg.get("role", "")
                msg["content"] = (
                    "[content removed]" if role == "assistant"
                    else " "  # minimal valid user content
                )
                fixes += 1

        elif isinstance(content, list):
            # Remove empty text blocks
            original_len = len(content)
            content[:] = [
                b for b in content
                if not (
                    isinstance(b, dict)
                    and b.get("type") == "text"
                    and not b.get("text", "").strip()
                )
            ]
            fixes += original_len - len(content)

            # If content list is now empty, add a minimal valid block
            if not content:
                role = msg.get("role", "")
                content.append({
                    "type": "text",
                    "text": (
                        "[content removed]" if role == "assistant"
                        else " "
                    ),
                })
                fixes += 1

            msg["content"] = content

    return fixes


def strip_response_headers(raw_headers) -> dict:
    """Strip hop-by-hop headers that shouldn't be forwarded."""
    skip = {
        "transfer-encoding",
        "content-length",
        "content-encoding",
        "connection",
        "keep-alive",
    }
    return {k: v for k, v in raw_headers.items() if k.lower() not in skip}
