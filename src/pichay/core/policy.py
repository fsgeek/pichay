from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from pichay.core.models import CanonicalRequest, PolicyAction, PolicyContext
from pichay.core.utils import content_bytes


@dataclass
class PolicyConfig:
    enable_paging: bool = True
    enable_trim: bool = True
    min_evict_size: int = 500


@dataclass
class BlockRef:
    target_id: str
    message_index: int
    block_index: int
    block: dict[str, Any]
    size_bytes: int
    text: str


def _block_text(block: dict[str, Any]) -> str:
    if isinstance(block.get("text"), str):
        return block["text"]
    if isinstance(block.get("content"), str):
        return block["content"]
    return ""


def collect_blocks(req: CanonicalRequest) -> list[BlockRef]:
    refs: list[BlockRef] = []
    for mi, msg in enumerate(req.messages):
        for bi, block in enumerate(msg.content):
            text = _block_text(block)
            refs.append(
                BlockRef(
                    target_id=f"m{mi}:b{bi}",
                    message_index=mi,
                    block_index=bi,
                    block=block,
                    size_bytes=content_bytes(block),
                    text=text,
                )
            )
    return refs


def phantom_stage(req: CanonicalRequest, ctx: PolicyContext) -> tuple[PolicyContext, list[PolicyAction]]:
    actions: list[PolicyAction] = []
    for ref in collect_blocks(req):
        if ref.block.get("pichay_phantom_protected") is True:
            ctx.protected_targets.add(ref.target_id)
    return ctx, actions


def paging_stage(req: CanonicalRequest, ctx: PolicyContext, cfg: PolicyConfig) -> list[PolicyAction]:
    if not cfg.enable_paging:
        return []

    refs = collect_blocks(req)
    counts = Counter(r.text for r in refs if r.text)
    actions: list[PolicyAction] = []

    for ref in refs:
        if ref.size_bytes < cfg.min_evict_size:
            continue
        btype = ref.block.get("type")
        if btype not in {"tool_result", "text"}:
            continue
        if counts.get(ref.text, 0) < 2:
            continue
        actions.append(
            PolicyAction(
                stage="paging",
                action="evict",
                target_id=ref.target_id,
                message_index=ref.message_index,
                block_index=ref.block_index,
                replacement_text=(
                    f"[Paged out duplicate block: {ref.size_bytes} bytes]"
                ),
                bytes=ref.size_bytes,
                duplication_score=float(counts[ref.text]),
            )
        )
    return actions


def trim_stage(req: CanonicalRequest, ctx: PolicyContext, cfg: PolicyConfig) -> list[PolicyAction]:
    if not cfg.enable_trim:
        return []

    refs = collect_blocks(req)
    seen: set[tuple[str, str]] = set()
    actions: list[PolicyAction] = []
    for ref in refs:
        key = (req.messages[ref.message_index].role, ref.text)
        if not ref.text:
            continue
        if key in seen:
            actions.append(
                PolicyAction(
                    stage="trim",
                    action="trim_duplicate",
                    target_id=ref.target_id,
                    message_index=ref.message_index,
                    block_index=ref.block_index,
                    replacement_text="[Trimmed duplicate block]",
                    bytes=ref.size_bytes,
                    duplication_score=1.0,
                )
            )
        else:
            seen.add(key)
    return actions


def apply_action(req: CanonicalRequest, action: PolicyAction) -> bool:
    try:
        msg = req.messages[action.message_index]
        block = msg.content[action.block_index]
    except (IndexError, KeyError):
        return False

    if action.replacement_text is None:
        return False

    block_type = block.get("type", "text")
    if block_type == "tool_result":
        # Anthropic tool_result blocks use `content`, not `text`.
        block["type"] = block_type
        block.pop("text", None)
        block["content"] = action.replacement_text
        return True
    if block_type == "text":
        block["type"] = block_type
        block["text"] = action.replacement_text
        return True

    msg.content[action.block_index] = {
        "type": "text",
        "text": action.replacement_text,
    }
    return True
