"""Phantom tools — side-channel communication between proxy and model.

The proxy injects phantom tools into the request. The model can call
them to communicate memory management hints. The proxy intercepts
these calls from the SSE stream before the framework sees them.

The framework never knows. The model and the proxy have a private
channel.

Tools:
    memory_fault(paths): Model requests evicted content back.
        The proxy resolves from PageStore, no file system round trip.

Note: memory_release is handled via inline <memory_cleanup> tags
(see tags.py) or by the framework's native memory_release tool.
It no longer needs phantom tool infrastructure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


class CleanupTagFilter:
    """Strips <memory_cleanup>...</memory_cleanup> tags from SSE text streams
    and executes the cleanup operations inline.

    Tags can span multiple text_delta chunks. The filter buffers pending
    text when it sees a partial tag opening, accumulates the tag body,
    executes operations when the closing tag arrives, and re-emits clean
    text outside tags.

    Pass block_store and page_store to execute operations as tags are
    stripped. Without them, the filter only strips (no execution).
    """

    _OPEN = "<memory_cleanup>"
    _CLOSE = "</memory_cleanup>"

    def __init__(self, block_store=None, page_store=None) -> None:
        self._inside = False
        self._pending = ""  # text that might be the start of a tag
        self._tag_body: list[str] = []  # accumulates content inside tag
        self._bs = block_store
        self._ps = page_store
        self.executed_ops: list[str] = []  # log of executed operations

    def _execute_tag_body(self) -> None:
        """Parse and execute cleanup operations from the accumulated tag body."""
        from pichay.tags import parse_cleanup_tags
        body = "".join(self._tag_body)
        self._tag_body = []
        ops = parse_cleanup_tags(f"<memory_cleanup>{body}</memory_cleanup>")
        if ops.empty:
            return
        if self._bs is not None:
            for block_id in ops.drops:
                if self._bs.drop(block_id):
                    self.executed_ops.append(f"dropped {block_id}")
            for block_id, summary in ops.summaries:
                if self._bs.summarize(block_id, summary):
                    self.executed_ops.append(f"summarized {block_id}")
            for block_id in ops.anchors:
                if self._bs.anchor(block_id):
                    self.executed_ops.append(f"anchored {block_id}")
        if self._ps is not None and ops.releases:
            for path in ops.releases:
                self._ps.mark_released(path)
            self.executed_ops.append(f"released {len(ops.releases)} path(s)")

    def filter(self, text: str) -> str:
        """Filter cleanup tags from a text chunk. Returns clean text."""
        result = []

        i = 0
        while i < len(text):
            if self._inside:
                # Look for closing tag
                close_pos = text.find(self._CLOSE, i)
                if close_pos >= 0:
                    # Accumulate body up to the closing tag, then execute
                    self._tag_body.append(text[i:close_pos])
                    self._execute_tag_body()
                    self._inside = False
                    i = close_pos + len(self._CLOSE)
                    # Skip optional newline after closing tag
                    if i < len(text) and text[i] == "\n":
                        i += 1
                else:
                    # Entire remaining text is inside tag — accumulate
                    self._tag_body.append(text[i:])
                    break

            elif self._pending:
                # We have buffered text that might be a partial open tag
                combined = self._pending + text[i:]
                if self._OPEN in combined:
                    # Tag completed — emit everything before it
                    tag_pos = combined.find(self._OPEN)
                    result.append(combined[:tag_pos])
                    self._pending = ""
                    self._inside = True
                    i = tag_pos + len(self._OPEN) - len(self._pending)
                    # Recalculate: skip past the open tag in the original text
                    consumed_from_text = len(combined) - len(text) + i
                    i = max(0, consumed_from_text)
                    continue
                elif self._OPEN.startswith(combined):
                    # Still a partial match — keep buffering
                    self._pending = combined
                    break
                else:
                    # False alarm — emit pending and restart
                    result.append(self._pending)
                    self._pending = ""
                    # Don't advance i — reprocess from same position

            else:
                # Normal state — look for tag opening
                open_pos = text.find(self._OPEN, i)
                partial_start = self._find_partial_open(text, i)

                if open_pos >= 0 and (partial_start < 0 or open_pos <= partial_start):
                    result.append(text[i:open_pos])
                    self._inside = True
                    i = open_pos + len(self._OPEN)
                elif partial_start >= 0:
                    # Partial tag at end of chunk — buffer it
                    result.append(text[i:partial_start])
                    self._pending = text[partial_start:]
                    break
                else:
                    result.append(text[i:])
                    break

        return "".join(result)

    def _find_partial_open(self, text: str, start: int) -> int:
        """Find position of a partial <memory_cleanup> tag at end of text."""
        tag = self._OPEN
        for length in range(len(tag) - 1, 0, -1):
            if text.endswith(tag[:length], start):
                pos = len(text) - length
                if pos >= start:
                    return pos
        return -1

    def flush(self) -> str:
        """Return any buffered text that turned out not to be a tag."""
        pending = self._pending
        self._pending = ""
        # If we're still inside a tag at flush, that's a malformed tag — discard
        self._inside = False
        return pending

PHANTOM_TOOL_NAMES = frozenset({"yuyay", "recall", "memory_fault", "qunqay", "tiqsiy"})

PHANTOM_TOOL_DEFINITIONS = [
    {
        "name": "yuyay",
        "description": (
            "Remember — restore evicted content by tensor handle. "
            "When you see '[tensor:xxxx — description]' markers in "
            "your context, the original content has been evicted to "
            "save space. Call this with the tensor handle(s) to get "
            "the content back. Faster and cheaper than re-reading "
            "files. You can also pass file paths or tool_use_ids "
            "for backward compatibility."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "handles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tensor handles (e.g. 'a3f2b901'), file paths, "
                        "or tool_use_ids to restore from eviction cache"
                    ),
                }
            },
            "required": ["handles"],
        },
    },
    {
        "name": "qunqay",
        "description": (
            "Release — mark content for eviction from the working set. "
            "Use this to proactively free context space by releasing "
            "tensors, file reads, or tool results you no longer need. "
            "The content remains in the backing store and can be "
            "restored later with yuyay. Use this when: (1) you rewrote "
            "a file and the old read is stale, (2) a tool result has "
            "been fully consumed, (3) you want to free space for "
            "content you value more. Every token you release saves "
            "O(n^2) compute on every subsequent turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "handles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tensor handles, file paths, or tool_use_ids "
                        "to release from the working set"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why this content is being released (logged "
                        "for audit trail)"
                    ),
                },
            },
            "required": ["handles"],
        },
    },
    {
        "name": "tiqsiy",
        "description": (
            "Compact — structurally compress older conversation turns. "
            "Replaces a range of older turns with a single user/assistant "
            "summary pair you provide. Original turns are archived to "
            "the backing store. Use this when conversation history is "
            "consuming context but you've captured the important "
            "conclusions. This reduces both content tokens and structural "
            "overhead (role labels, turn boundaries). The compaction is "
            "applied on the next turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "older_than": {
                    "type": "integer",
                    "description": (
                        "Compact messages more than this many turns "
                        "from the current turn. Messages newer than "
                        "this are preserved."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Your summary of the compacted conversation "
                        "segment. Should capture conclusions, decisions, "
                        "and any context needed for future reference. "
                        "This becomes the content of the replacement "
                        "assistant message."
                    ),
                },
            },
            "required": ["older_than", "summary"],
        },
    },
]


@dataclass
class PhantomCall:
    """A captured phantom tool call from the model's response."""

    name: str
    tool_use_id: str
    input: dict = field(default_factory=dict)


def inject_tools(body: dict) -> set[str]:
    """Add phantom tool definitions to the request's tools array.

    Returns a set of tool names that the framework already provides
    (observe-only). Tools NOT in this set are fully intercepted by
    the gateway — their events are suppressed from the stream and
    handled via continuation.

    Returns empty set when no phantom tools are framework-provided
    (all are intercepted).
    """
    tools = body.get("tools", [])
    existing_names = {t.get("name") for t in tools}
    observe_only = PHANTOM_TOOL_NAMES & existing_names
    # Inject definitions for tools the framework doesn't provide
    for defn in PHANTOM_TOOL_DEFINITIONS:
        if defn["name"] not in existing_names:
            tools.append(defn)
    body["tools"] = tools
    return observe_only


def inject_phantom_results(
    messages: list[dict],
    phantom_calls: list[PhantomCall],
    page_store,
    observe_only: bool | set[str] = False,
) -> list[dict]:
    """Inject tool_result messages for phantom calls from the previous turn.

    The model called phantom tools, but the framework never saw them.
    We need to inject the results so the model's next turn sees a
    coherent conversation — it called a tool, it got a result.

    For memory_fault calls, the result includes the restored content.

    observe_only can be:
    - True: skip all injection (framework handled everything)
    - False: inject all (gateway intercepted everything)
    - set[str]: skip injection for tools in set (framework handled
      those); inject for tools NOT in set (gateway intercepted those)
    """
    if not phantom_calls:
        return messages
    if observe_only is True:
        return messages

    # Filter to only intercepted calls (not framework-handled)
    if isinstance(observe_only, set) and observe_only:
        phantom_calls = [
            c for c in phantom_calls if c.name not in observe_only
        ]
        if not phantom_calls:
            return messages

    # Find the last assistant message — that's where the phantom calls were
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return messages

    # Re-insert the phantom tool_use blocks into the assistant message
    assistant_msg = messages[last_assistant_idx]
    content = assistant_msg.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]

    existing_tool_ids = {
        b.get("id") for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    }
    for call in phantom_calls:
        if call.tool_use_id in existing_tool_ids:
            continue  # already present — don't duplicate
        content.append(
            {
                "type": "tool_use",
                "id": call.tool_use_id,
                "name": call.name,
                "input": call.input,
            }
        )
    assistant_msg["content"] = content

    # Build tool_result messages for each phantom call
    results = []
    for call in phantom_calls:
        result_content = _handle_phantom_call(call, page_store)
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": call.tool_use_id,
                "content": result_content,
            }
        )

    # Find where to inject — after the assistant message's tool results
    # The next user message (if any) should contain the results
    inject_idx = last_assistant_idx + 1
    if inject_idx < len(messages) and messages[inject_idx].get("role") == "user":
        # Append to existing user message, skipping duplicates
        user_msg = messages[inject_idx]
        user_content = user_msg.get("content", [])
        if isinstance(user_content, str):
            user_content = [{"type": "text", "text": user_content}]
        existing_result_ids = {
            b.get("tool_use_id") for b in user_content
            if isinstance(b, dict) and b.get("type") == "tool_result"
        }
        for r in results:
            if r["tool_use_id"] not in existing_result_ids:
                user_content.append(r)
        user_msg["content"] = user_content
    else:
        # Insert a new user message with the results
        messages.insert(inject_idx, {"role": "user", "content": results})

    return messages


def _handle_phantom_call(call: PhantomCall, page_store,
                         block_store=None) -> str:
    """Execute a phantom tool call and return the result text."""
    if call.name == "qunqay":
        identifiers = call.input.get("handles", [])
        reason = call.input.get("reason", "model requested release")
        released = []
        not_found = []
        for identifier in identifiers:
            found = False
            # Try page store (file reads, tool results)
            if page_store is not None:
                if page_store.mark_released(identifier):
                    found = True
                # Also try eviction index (file paths)
                elif identifier in getattr(page_store, '_eviction_index', {}):
                    page_store.mark_released(identifier)
                    found = True
            # Try block store (conversation blocks)
            if not found and block_store is not None:
                if block_store.drop(identifier):
                    found = True
            if found:
                released.append(identifier)
            else:
                not_found.append(identifier)
        parts = []
        if released:
            parts.append(f"Released {len(released)} tensor(s): {', '.join(released)}")
        if not_found:
            parts.append(f"Not found: {', '.join(not_found)}")
        if reason:
            parts.append(f"Reason: {reason}")
        return " | ".join(parts) if parts else "Nothing to release."

    if call.name in ("yuyay", "recall", "memory_fault"):
        # Accept both "handles" (new) and "paths" (legacy) parameter names
        identifiers = call.input.get("handles", call.input.get("paths", []))
        restored = []
        not_found = []
        for identifier in identifiers:
            entry = None

            # 1. Try tensor handle (unified addressing)
            if page_store is not None:
                entry = page_store.resolve_tensor(identifier)
            if entry is None and block_store is not None:
                content = block_store.restore(identifier)
                if content is not None:
                    restored.append({
                        "label": f"tensor:{identifier}",
                        "content": content,
                        "size": len(content),
                    })
                    continue

            # 2. Try file path (Read results)
            if entry is None and page_store is not None:
                entry = page_store._eviction_index.get(identifier)

            # 3. Try tool_use_id (legacy)
            if entry is None and page_store is not None:
                entry = page_store.pages.get(identifier)

            if entry is not None:
                label = f"tensor:{identifier}"
                if entry.tool_name == "Read":
                    path = entry.tool_input.get("file_path", identifier)
                    label = f"tensor:{identifier} ({path})"
                elif entry.tool_name:
                    label = f"tensor:{identifier} ({entry.tool_name})"
                restored.append(
                    {
                        "label": label,
                        "content": entry.original_content,
                        "size": entry.original_size,
                    }
                )
            else:
                not_found.append(identifier)

        parts = []
        for r in restored:
            parts.append(f"--- {r['label']} ({r['size']} bytes) ---\n{r['content']}")
        if not_found:
            parts.append(
                f"Not in cache (use Read or re-run): {', '.join(not_found)}"
            )
        return "\n\n".join(parts) if parts else "No handles requested."

    if call.name == "tiqsiy":
        # Structural compaction is deferred — we store the request
        # in the session and apply it on the next inbound request.
        # The handler in proxy.py reads pending_compaction from session.
        older_than = call.input.get("older_than", 20)
        summary = call.input.get("summary", "")
        if not summary:
            return "Error: summary is required for compaction."
        # Return confirmation — actual compaction happens next turn.
        return (
            f"Compaction scheduled: messages older than {older_than} turns "
            f"will be replaced with your summary on the next turn. "
            f"Summary length: {len(summary)} chars."
        )

    return f"Unknown phantom tool: {call.name}"


@dataclass
class CompactionResult:
    """Result of a structural compaction operation."""

    messages_removed: int = 0
    messages_after: int = 0
    chars_removed: int = 0
    archived: list[dict] = field(default_factory=list)


def apply_compaction(
    messages: list[dict],
    older_than: int,
    summary: str,
) -> CompactionResult:
    """Replace older conversation turns with a summary pair.

    Finds messages older than `older_than` turns from the end,
    replaces them with a single user/assistant summary pair.
    Returns the archived messages for storage.

    Preserves the first message (often contains system context
    from the framework) and maintains valid alternating structure.
    """
    result = CompactionResult()
    total = len(messages)

    if total < 4:  # need at least 2 turns to compact
        return result

    # Count turns (pairs of user/assistant messages)
    # Each turn is roughly 2 messages
    preserve_count = older_than * 2
    if preserve_count >= total:
        return result  # nothing old enough to compact

    # Split: archive the old, keep the recent
    # Always preserve the first message (framework context)
    compact_end = total - preserve_count
    if compact_end <= 1:
        return result

    archived = messages[1:compact_end]  # skip first message
    preserved_first = messages[0]
    preserved_recent = messages[compact_end:]

    # Calculate what we're removing
    archived_chars = sum(
        len(json.dumps(m).encode("utf-8")) for m in archived
    )

    # Build the summary pair — a single user/assistant exchange
    # that replaces the archived conversation segment
    summary_pair = [
        {
            "role": "user",
            "content": (
                f"[Compacted: {len(archived)} messages from earlier "
                f"in this conversation were structurally compressed "
                f"into this summary by the model.]"
            ),
        },
        {
            "role": "assistant",
            "content": summary,
        },
    ]

    # Ensure valid alternation after first message
    # The first preserved message sets the expected next role
    new_messages = [preserved_first] + summary_pair + preserved_recent

    # Validate alternation — fix if needed
    _fix_alternation(new_messages)

    result.messages_removed = len(archived)
    result.messages_after = len(new_messages)
    result.chars_removed = archived_chars
    result.archived = archived

    # Replace in-place
    messages.clear()
    messages.extend(new_messages)

    return result


def _fix_alternation(messages: list[dict]) -> None:
    """Ensure strict user/assistant alternation.

    If two consecutive messages have the same role, merge the
    second into the first.
    """
    i = 1
    while i < len(messages):
        if messages[i].get("role") == messages[i - 1].get("role"):
            # Merge: append content of messages[i] to messages[i-1]
            prev = messages[i - 1]
            curr = messages[i]
            prev_content = prev.get("content", "")
            curr_content = curr.get("content", "")
            if isinstance(prev_content, str) and isinstance(curr_content, str):
                prev["content"] = prev_content + "\n\n" + curr_content
            elif isinstance(prev_content, list) and isinstance(curr_content, list):
                prev["content"] = prev_content + curr_content
            elif isinstance(prev_content, str):
                prev["content"] = [
                    {"type": "text", "text": prev_content}
                ] + (curr_content if isinstance(curr_content, list) else [
                    {"type": "text", "text": str(curr_content)}
                ])
            else:
                prev["content"] = prev_content + [
                    {"type": "text", "text": str(curr_content)}
                ]
            messages.pop(i)
        else:
            i += 1


def filtered_stream(
    byte_iter,
    collected_chunks,
    phantom_calls_out,
    observe_only: bool | set[str] = False,
    block_store=None,
    page_store=None,
    session_id: str = "",
    continuation_needed: list | None = None,
):
    """Filter phantom tool events from an SSE byte stream.

    Yields filtered bytes to the framework. Phantom tool_use events
    are suppressed. Completed phantom calls are appended to
    phantom_calls_out.

    When ALL tool_use blocks in a response are phantom, the stop_reason
    is rewritten from "tool_use" to "end_turn" so the framework doesn't
    enter tool execution mode for tools it never saw.

    observe_only can be:
    - False: all phantom tools are intercepted (suppress from stream)
    - True: all phantom tools are observed only (legacy compat)
    - set[str]: tools in the set are observed; others are intercepted

    collected_chunks receives ALL bytes (including phantom) for logging.
    """
    # Normalize observe_only to a set of tool names
    if observe_only is True:
        _observe_set = PHANTOM_TOOL_NAMES  # all are observe-only
    elif observe_only is False:
        _observe_set: set[str] = set()  # none are observe-only
    else:
        _observe_set = observe_only  # per-tool set

    phantom_block_indices: set[int] = set()
    real_tool_indices: set[int] = set()
    # Track which phantom blocks are intercept-only (not observe-only)
    intercept_block_indices: set[int] = set()
    phantom_building: dict[int, dict] = {}
    cleanup_filter = CleanupTagFilter(block_store=block_store, page_store=page_store)
    if continuation_needed is None:
        continuation_needed = []
    buffer = b""

    for chunk in byte_iter:
        collected_chunks.append(chunk)
        buffer += chunk

        # Process complete SSE events (delimited by \n\n)
        while b"\n\n" in buffer:
            event_bytes, buffer = buffer.split(b"\n\n", 1)
            event_text = event_bytes.decode("utf-8", errors="replace")

            suppress = False
            data_str = None

            for line in event_text.split("\n"):
                if line.startswith("data: "):
                    data_str = line[6:]

            # Suppress [DONE] when continuation is pending —
            # the continuation will provide its own [DONE]
            if data_str == "[DONE]" and continuation_needed:
                continue

            if data_str and data_str != "[DONE]":
                try:
                    event_data = json.loads(data_str)

                    # Strip cleanup tags from text_delta events before
                    # the framework sees them. Tags can span chunks so
                    # CleanupTagFilter maintains state across events.
                    # Only strip if we have any intercept-mode tools active.
                    if (
                        _observe_set != PHANTOM_TOOL_NAMES
                        and event_data.get("type") == "content_block_delta"
                    ):
                        delta = event_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            raw_text = delta.get("text", "")
                            clean_text = cleanup_filter.filter(raw_text)
                            if clean_text != raw_text:
                                delta["text"] = clean_text
                                event_data["delta"] = delta
                                data_str = json.dumps(event_data)
                                event_bytes = f"data: {data_str}".encode("utf-8")

                    suppress = _classify_event(
                        event_data,
                        phantom_block_indices,
                        real_tool_indices,
                        phantom_building,
                        phantom_calls_out,
                        intercept_indices=intercept_block_indices,
                        observe_set=_observe_set,
                    )
                    # When all content blocks were intercepted phantom
                    # tools (no real tools, no observe-only phantom tools),
                    # the gateway needs to auto-continue.
                    if (
                        intercept_block_indices
                        and event_data.get("type") == "message_delta"
                        and not real_tool_indices
                        # Only continue if ALL phantom blocks are intercepted
                        and phantom_block_indices == intercept_block_indices
                    ):
                        delta = event_data.get("delta", {})
                        if delta.get("stop_reason") == "tool_use":
                            # Signal continuation needed
                            continuation_needed.append(True)
                            # Suppress this event — continuation will
                            # provide its own stop events
                            suppress = True
                except json.JSONDecodeError:
                    pass

            # Also suppress message_stop when continuation is pending
            if (
                not suppress
                and continuation_needed
                and data_str and data_str != "[DONE]"
            ):
                try:
                    evt = json.loads(data_str)
                    if evt.get("type") == "message_stop":
                        suppress = True
                except json.JSONDecodeError:
                    pass

            if not suppress:
                yield event_bytes + b"\n\n"

    # Flush remaining buffer
    if buffer:
        yield buffer

    # Flush cleanup filter — discard any partial/unclosed tag
    cleanup_filter.flush()
    if cleanup_filter.executed_ops:
        import sys
        print(
            f"  [{session_id}] CLEANUP (stream): {'; '.join(cleanup_filter.executed_ops)}",
            file=sys.stderr,
        )


def _classify_event(
    event_data: dict,
    phantom_indices: set[int],
    real_tool_indices: set[int],
    building: dict[int, dict],
    completed: list[PhantomCall],
    intercept_indices: set[int] | None = None,
    observe_set: set[str] | None = None,
) -> bool:
    """Returns True if this event should be suppressed.

    When observe_set is provided, only tools NOT in the set are
    suppressed (intercept mode). Tools in the set are captured
    but not suppressed (observe mode).
    """
    if intercept_indices is None:
        intercept_indices = set()
    if observe_set is None:
        observe_set = set()

    event_type = event_data.get("type", "")
    index = event_data.get("index")

    if event_type == "content_block_start":
        cb = event_data.get("content_block", {})
        if cb.get("type") == "tool_use":
            tool_name = cb.get("name", "")
            if tool_name in PHANTOM_TOOL_NAMES:
                phantom_indices.add(index)
                building[index] = {
                    "name": tool_name,
                    "id": cb.get("id", ""),
                    "input_json": "",
                }
                if tool_name not in observe_set:
                    intercept_indices.add(index)
                    return True  # suppress — we're intercepting
                return False  # observe — let it through
            else:
                # Real tool — track so we know stop_reason is legitimate
                real_tool_indices.add(index)

    elif event_type == "content_block_delta" and index in phantom_indices:
        delta = event_data.get("delta", {})
        if delta.get("type") == "input_json_delta":
            building[index]["input_json"] += delta.get("partial_json", "")
        return index in intercept_indices

    elif event_type == "content_block_stop" and index in phantom_indices:
        if index in building:
            rec = building.pop(index)
            try:
                parsed_input = json.loads(rec["input_json"])
            except json.JSONDecodeError:
                parsed_input = {}
            completed.append(
                PhantomCall(
                    name=rec["name"],
                    tool_use_id=rec["id"],
                    input=parsed_input,
                )
            )
        return index in intercept_indices

    return False
