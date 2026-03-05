"""Stateless message manipulation helpers.

These functions were extracted from proxy.py's create_app() closure.
They don't use any closure state — all dependencies are explicit parameters.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pichay.blocks import BlockStore

PICHAY_STATUS_MARKER = "[pichay-system-status]"


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
    from pichay.tags import parse_cleanup_tags, strip_cleanup_tags

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
                msg["content"] = strip_cleanup_tags(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                ops = parse_cleanup_tags(text)
                if not ops.empty:
                    total_ops.append(ops)
                    block["text"] = strip_cleanup_tags(text)

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

    return "; ".join(stats_parts) if stats_parts else None


def inject_system_status(body: dict, ts: dict, cap: int,
                         request_time, block_store=None) -> None:
    """Inject or replace a system status block with context pressure info.

    Gives the model awareness of:
    - That it's running under Pichay (experimental virtual memory)
    - Current time/date
    - Memory usage relative to the hard cap (85% of context window)
    - Block inventory and cooperative memory instructions at moderate+ pressure
    """
    effective = ts["last_effective"]
    context_limit = cap if cap > 0 else 200_000
    hard_cap = int(context_limit * 0.85)
    pct = (effective / context_limit * 100) if context_limit > 0 else 0
    pressure = "low"
    if pct > 70:
        pressure = "high"
    elif pct > 50:
        pressure = "moderate"

    time_str = request_time.strftime("%Y-%m-%d %H:%M:%S UTC")

    status_text = (
        f"{PICHAY_STATUS_MARKER}\n"
        f"This system is running under Pichay, an experimental "
        f"virtual memory manager for LLM context windows. Tool "
        f"results may be evicted and replaced with [Paged out: ...] "
        f"summaries. You can restore evicted content using the "
        f"memory_fault tool (pass file paths or tool_use_ids). "
        f"You can proactively release content you no longer need "
        f"using memory_release. If you observe anomalous behavior "
        f"(missing context, unexpected gaps, incoherent references), "
        f"describe it to aid debugging.\n\n"
        f"Current time: {time_str}\n"
        f"Context usage: {effective:,} / {context_limit:,} tokens "
        f"({pct:.0f}%) | pressure: {pressure}\n"
        f"Hard cap: {hard_cap:,} tokens (85% of context window)"
    )

    if pressure in ("moderate", "high") and block_store is not None:
        large = block_store.large_blocks(min_size=2000)
        if large:
            block_lines = []
            for b in large[:5]:
                size_k = b.size / 1024
                block_lines.append(
                    f"  - [block:{b.block_id}] {b.role} turn {b.turn} "
                    f"({size_k:.1f}KB): {b.preview}"
                )
            status_text += (
                f"\n\nLargest conversation blocks "
                f"({block_store.block_count} tracked):\n"
                + "\n".join(block_lines)
            )

        status_text += (
            "\n\nCooperative memory: You can manage conversation memory "
            "by including <memory_cleanup> tags in your response. "
            "Operations:\n"
            "  drop: block:XXXX           — archive block content\n"
            "  summarize: block:XXXX \"summary\" — replace with your summary\n"
            "  anchor: block:XXXX         — mark block to keep\n"
            "  release: path1, path2      — release file contents\n"
            "Tags are processed on the next turn and stripped from history."
        )

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
