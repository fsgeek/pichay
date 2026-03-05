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
from dataclasses import dataclass, field

PHANTOM_TOOL_NAMES = frozenset({"memory_fault"})

PHANTOM_TOOL_DEFINITIONS = [
    {
        "name": "memory_fault",
        "description": (
            "Request evicted content back from the memory manager. "
            "When you see a '[Paged out: ...]' marker for content you "
            "need, call this instead of re-reading the file or re-running "
            "the command. The memory manager will restore the content "
            "directly — faster and cheaper than a file system read. You "
            "can request multiple items at once. Pass file paths for "
            "Read results, or tool_use_ids (from the paged-out marker) "
            "for Bash, Grep, Agent, and other non-file results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "File paths to restore from eviction cache, "
                        "or tool_use_ids for non-file results"
                    ),
                }
            },
            "required": ["paths"],
        },
    },
]


@dataclass
class PhantomCall:
    """A captured phantom tool call from the model's response."""

    name: str
    tool_use_id: str
    input: dict = field(default_factory=dict)


def inject_tools(body: dict) -> bool:
    """Add phantom tool definitions to the request's tools array.

    Returns True (observe_only) if the framework already provides tools
    with the same names — we observe phantom calls but don't strip them
    from the SSE stream to avoid 400 "tool use concurrency" errors.

    Returns False when phantom tools are injected and should be
    intercepted from the stream.
    """
    tools = body.get("tools", [])
    existing_names = {t.get("name") for t in tools}
    framework_provides = PHANTOM_TOOL_NAMES & existing_names
    if framework_provides:
        return True
    for defn in PHANTOM_TOOL_DEFINITIONS:
        tools.append(defn)
    body["tools"] = tools
    return False


def inject_phantom_results(
    messages: list[dict],
    phantom_calls: list[PhantomCall],
    page_store,
    observe_only: bool = False,
) -> list[dict]:
    """Inject tool_result messages for phantom calls from the previous turn.

    The model called phantom tools, but the framework never saw them.
    We need to inject the results so the model's next turn sees a
    coherent conversation — it called a tool, it got a result.

    For memory_fault calls, the result includes the restored content.

    When observe_only is True, the framework already handled these tools,
    so we skip message injection entirely.
    """
    if not phantom_calls:
        return messages
    if observe_only:
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

    for call in phantom_calls:
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
        # Append to existing user message
        user_msg = messages[inject_idx]
        user_content = user_msg.get("content", [])
        if isinstance(user_content, str):
            user_content = [{"type": "text", "text": user_content}]
        user_content.extend(results)
        user_msg["content"] = user_content
    else:
        # Insert a new user message with the results
        messages.insert(inject_idx, {"role": "user", "content": results})

    return messages


def _handle_phantom_call(call: PhantomCall, page_store,
                         block_store=None) -> str:
    """Execute a phantom tool call and return the result text."""
    if call.name == "memory_fault":
        paths = call.input.get("paths", [])
        restored = []
        not_found = []
        if page_store is not None:
            for identifier in paths:
                # Try file path first (Read results)
                entry = page_store._eviction_index.get(identifier)
                if entry is None:
                    # Try tool_use_id (Bash, Agent, Grep, etc.)
                    entry = page_store.pages.get(identifier)
                if entry is None and block_store is not None:
                    # Try block ID (conversation blocks)
                    content = block_store.restore(identifier)
                    if content is not None:
                        restored.append({
                            "label": f"block {identifier}",
                            "content": content,
                            "size": len(content),
                        })
                        continue
                if entry is not None:
                    label = identifier
                    if entry.tool_name and entry.tool_name != "Read":
                        label = f"{entry.tool_name} {identifier}"
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
        return "\n\n".join(parts) if parts else "No paths requested."

    return f"Unknown phantom tool: {call.name}"


def filtered_stream(
    byte_iter,
    collected_chunks,
    phantom_calls_out,
    observe_only: bool = False,
):
    """Filter phantom tool events from an SSE byte stream.

    Yields filtered bytes to the framework. Phantom tool_use events
    are suppressed. Completed phantom calls are appended to
    phantom_calls_out.

    When ALL tool_use blocks in a response are phantom, the stop_reason
    is rewritten from "tool_use" to "end_turn" so the framework doesn't
    enter tool execution mode for tools it never saw.

    When observe_only is True, phantom events are captured but not
    suppressed from the stream.

    collected_chunks receives ALL bytes (including phantom) for logging.
    """
    phantom_block_indices: set[int] = set()
    real_tool_indices: set[int] = set()
    phantom_building: dict[int, dict] = {}
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

            if data_str and data_str != "[DONE]":
                try:
                    event_data = json.loads(data_str)
                    suppress = _classify_event(
                        event_data,
                        phantom_block_indices,
                        real_tool_indices,
                        phantom_building,
                        phantom_calls_out,
                    )
                    # When all content blocks were phantom, the framework
                    # would receive an empty response. Inject a synthetic
                    # text block and rewrite stop_reason before message_delta.
                    if (
                        not observe_only
                        and event_data.get("type") == "message_delta"
                        and phantom_block_indices
                        and not real_tool_indices
                    ):
                        delta = event_data.get("delta", {})
                        if delta.get("stop_reason") == "tool_use":
                            delta["stop_reason"] = "end_turn"
                            # Inject synthetic text block so response isn't empty
                            synthetic_start = json.dumps({
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""},
                            })
                            synthetic_delta = json.dumps({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": "[released from context]"},
                            })
                            synthetic_stop = json.dumps({
                                "type": "content_block_stop",
                                "index": 0,
                            })
                            yield f"data: {synthetic_start}\n\n".encode()
                            yield f"data: {synthetic_delta}\n\n".encode()
                            yield f"data: {synthetic_stop}\n\n".encode()
                            rewritten = "data: " + json.dumps(event_data)
                            event_bytes = rewritten.encode("utf-8")
                except json.JSONDecodeError:
                    pass

            if not suppress or observe_only:
                yield event_bytes + b"\n\n"

    # Flush remaining buffer
    if buffer:
        yield buffer


def _classify_event(
    event_data: dict,
    phantom_indices: set[int],
    real_tool_indices: set[int],
    building: dict[int, dict],
    completed: list[PhantomCall],
) -> bool:
    """Returns True if this event should be suppressed."""
    event_type = event_data.get("type", "")
    index = event_data.get("index")

    if event_type == "content_block_start":
        cb = event_data.get("content_block", {})
        if cb.get("type") == "tool_use":
            if cb.get("name") in PHANTOM_TOOL_NAMES:
                phantom_indices.add(index)
                building[index] = {
                    "name": cb["name"],
                    "id": cb.get("id", ""),
                    "input_json": "",
                }
                return True
            else:
                # Real tool — track so we know stop_reason is legitimate
                real_tool_indices.add(index)

    elif event_type == "content_block_delta" and index in phantom_indices:
        delta = event_data.get("delta", {})
        if delta.get("type") == "input_json_delta":
            building[index]["input_json"] += delta.get("partial_json", "")
        return True

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
        return True

    return False
