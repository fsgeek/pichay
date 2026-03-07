#!/usr/bin/env python3
"""Logging proxy for Claude API calls, with optional context paging.

Two modes:
    Observe (default):  Logs request/response metrics. Pure observation.
    Compact (--compact): Also evicts stale tool results from the messages
        array before forwarding, replacing them with compact summaries.

Usage:
    # Observation only
    python -m pichay.proxy [--port 0] [--log-dir logs]

    # With context paging
    python -m pichay.proxy --compact [--age-threshold 4] [--min-size 500]

    # Point Claude Code at it
    ANTHROPIC_BASE_URL=http://localhost:<port> claude

Port 0 (default) picks a random free port to avoid collisions.

Session isolation: Each distinct conversation (identified by a fingerprint
of the first user message) gets its own token state, page store, and
phantom call queue. This prevents cross-contamination when Claude Code
spawns concurrent subagents through the same proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from flask import Flask, Response, request

from pichay.blocks import BlockStore
from pichay.pager import PageStore, compact_messages, compact_conversation
from pichay.tags import parse_cleanup_tags, strip_cleanup_tags
from pichay.message_ops import (
    PICHAY_STATUS_MARKER,
    process_cleanup_tags,
    inject_system_status,
    measure_system_prompt,
    measure_messages,
    strip_response_headers,
)
from pichay.phantom import (
    PhantomCall,
    _handle_phantom_call,
    filtered_stream,
    inject_phantom_results,
    inject_tools,
)

DEFAULT_API_BASE = "https://api.anthropic.com"


def _phantom_continuation(
    body: dict,
    phantom_calls: list[PhantomCall],
    page_store,
    block_store,
    headers: dict,
    upstream_path: str,
    client: httpx.Client,
    chunks_collected: list,
    sid: str,
):
    """Gateway auto-continue: execute phantom tools and stream continuation.

    When the model calls phantom tools (yuyay, qunqay), the proxy handles
    them and immediately sends a continuation request with the tool results.
    The model continues generating without a turn boundary. The framework
    never sees the phantom tools — just one seamless response.

    Yields SSE bytes that extend the original response stream.
    """
    # Build the assistant message content with phantom tool_use blocks
    # (We need to reconstruct what the model actually said)
    assistant_content = []
    for pc in phantom_calls:
        assistant_content.append({
            "type": "tool_use",
            "id": pc.tool_use_id,
            "name": pc.name,
            "input": pc.input,
        })

    # Build tool_result messages
    tool_results = []
    for pc in phantom_calls:
        result_text = _handle_phantom_call(pc, page_store,
                                           block_store=block_store)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": pc.tool_use_id,
            "content": result_text,
        })

    # Continuation messages: original messages + assistant + user with results
    messages = list(body.get("messages", []))
    messages.append({"role": "assistant", "content": assistant_content})
    messages.append({"role": "user", "content": tool_results})

    # Build continuation request (same params, updated messages)
    cont_body = {**body, "messages": messages, "stream": True}

    cont_size = len(json.dumps(cont_body))
    print(
        f"  [{sid}] CONTINUATION: {len(phantom_calls)} phantom call(s), "
        f"sending {len(messages)} messages, {cont_size // 1024}KB",
        file=sys.stderr,
    )

    try:
        cont_resp = client.send(
            client.build_request(
                "POST", upstream_path,
                json=cont_body, headers=headers,
            ),
            stream=True,
        )

        # Stream continuation, suppressing message_start
        # (framework already received one from the first response)
        buffer = b""
        for chunk in cont_resp.iter_bytes():
            chunks_collected.append(chunk)
            buffer += chunk

            while b"\n\n" in buffer:
                event_bytes, buffer = buffer.split(b"\n\n", 1)
                event_text = event_bytes.decode("utf-8", errors="replace")

                data_str = None
                for line in event_text.split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:]

                suppress = False
                if data_str and data_str != "[DONE]":
                    try:
                        evt = json.loads(data_str)
                        # Suppress envelope events — the framework already
                        # has message_start/message_delta/message_stop from
                        # the original response.  The continuation is an
                        # internal gateway round-trip; only content_block
                        # events should leak through to the client.
                        if evt.get("type") in (
                            "message_start",
                            "message_delta",
                            "message_stop",
                        ):
                            suppress = True
                    except json.JSONDecodeError:
                        pass

                if not suppress:
                    yield event_bytes + b"\n\n"

        if buffer:
            yield buffer

        cont_resp.close()
    except Exception as e:
        print(f"  [{sid}] CONTINUATION STREAM ERROR: {e}", file=sys.stderr)
        raise


def _session_id(body: dict) -> str:
    """Derive a stable session fingerprint from the first user message.

    Different conversations have different first messages, so this
    is unique per conversation and stable across turns (the first
    message stays in the array for the conversation's lifetime).

    The system prompt is excluded because it contains dynamic content
    (timestamps, git status, cache blocks) that changes between requests.
    """
    messages = body.get("messages", [])
    first = json.dumps(messages[0], sort_keys=True) if messages else ""
    return hashlib.sha256(first.encode()).hexdigest()[:8]


def find_free_port() -> int:
    """Find a free port by binding to port 0 and reading the assignment."""
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # Try IPv6 loopback first (reduces collision space)
            s.bind(("::1", 0))
        except OSError:
            # Fall back to IPv4
            s.close()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s4:
                s4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s4.bind(("127.0.0.1", 0))
                return s4.getsockname()[1]
        return s.getsockname()[1]


def create_app(
    log_dir: Path,
    compact: bool = False,
    trim: bool = False,
    age_threshold: int = 4,
    min_size: int = 500,
    upstream: str = DEFAULT_API_BASE,
    token_cap: int = 0,
) -> Flask:
    """Create the proxy Flask app."""
    app = Flask(__name__)
    log_dir.mkdir(parents=True, exist_ok=True)

    # One log file per proxy session
    session_start = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"proxy_{session_start}.jsonl"

    _token_warning_threshold = int(token_cap * 0.8) if token_cap > 0 else 0

    # ANSI colors for terminal output
    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    if token_cap > 0:
        print(
            f"Token cap: {token_cap:,} "
            f"(warning at {_token_warning_threshold:,})",
            file=sys.stderr,
        )

    # --- Session-keyed state ---
    # Each conversation gets its own token state, page store, and phantom
    # call queue. This prevents cross-contamination when multiple Claude
    # Code conversations (including subagents) hit the proxy concurrently.
    _sessions: dict[str, dict] = {}

    def _get_session(body: dict) -> dict:
        """Get or create per-conversation session state."""
        sid = _session_id(body)
        if sid not in _sessions:
            page_log = log_dir / f"pages_{session_start}_{sid}.jsonl"
            ps = PageStore(log_path=page_log)
            _sessions[sid] = {
                "id": sid,
                "token_state": {
                    "last_effective": 0,
                    "blocked": False,
                    "turn": 0,
                    "calibrated": False,
                },
                "page_store": ps,
                "block_store": BlockStore() if compact else None,
                "phantom_pending": [],
                "observe_only": False,
            }
            print(
                f"  {_DIM}[{sid}] new session{_RESET}",
                file=sys.stderr,
            )
        return _sessions[sid]

    # Trimmer state (shared — it's stateless per-request, just caches patterns)
    trimmer = None
    if trim:
        from pichay.trimmer import SystemPromptTrimmer
        trimmer = SystemPromptTrimmer()
        print(
            "Trim mode: tool stubs + skill dedup + static tracking",
            file=sys.stderr,
        )

    if compact:
        print(
            f"Compact mode: age_threshold={age_threshold}, "
            f"min_size={min_size}",
            file=sys.stderr,
        )

    # Persistent HTTP client for forwarding
    client = httpx.Client(
        base_url=upstream,
        timeout=httpx.Timeout(300.0, connect=30.0),
    )
    if upstream != DEFAULT_API_BASE:
        print(f"Upstream: {upstream}", file=sys.stderr)

    def log_record(record: dict) -> None:
        """Append a record to the log file."""
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _check_token_cap(usage: dict, session: dict) -> None:
        """Check effective tokens against cap, update state, emit warnings."""
        ts = session["token_state"]
        sid = session["id"]
        effective = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        # Always track usage (needed for system status injection)
        ts["last_effective"] = effective
        if token_cap <= 0:
            return
        pct = (effective / token_cap * 100) if token_cap > 0 else 0

        if effective > token_cap:
            ts["blocked"] = True
            print(
                f"{_RED}  [{sid}] TOKEN CAP EXCEEDED: {effective:,} / {token_cap:,} "
                f"({pct:.0f}%) — next request will be blocked{_RESET}",
                file=sys.stderr,
            )
            log_record({
                "type": "token_cap",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session": sid,
                "action": "exceeded",
                "effective_tokens": effective,
                "token_cap": token_cap,
                "pct": round(pct, 1),
                "turn": ts["turn"],
            })
        elif effective > _token_warning_threshold:
            print(
                f"{_YELLOW}  [{sid}] TOKEN WARNING: {effective:,} / {token_cap:,} "
                f"({pct:.0f}%) — approaching cap{_RESET}",
                file=sys.stderr,
            )
            log_record({
                "type": "token_cap",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session": sid,
                "action": "warning",
                "effective_tokens": effective,
                "token_cap": token_cap,
                "pct": round(pct, 1),
                "turn": ts["turn"],
            })

    def _display_turn_status(usage: dict, session: dict) -> None:
        """Post-response status line with real token count."""
        ts = session["token_state"]
        sid = session["id"]
        ps = session["page_store"]
        effective = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        if effective == 0:
            return

        cap_str = ""
        if token_cap > 0:
            pct = effective / token_cap * 100
            cap_str = f"/{token_cap // 1000}k ({pct:.0f}%)"

        ev_str = ""
        if ps is not None:
            pin_str = f" pin {len(ps._pinned)}" if ps._pinned else ""
            ev_str = (
                f" | ev {ps.unique_evictions} gc {ps.gc_count}{pin_str}"
                f" | faults {len(ps.faults)}/{ps.unique_evictions}"
            )

        print(
            f"  [{sid}] [Turn {ts['turn']}] "
            f"{effective:,} tok{cap_str}{ev_str}",
            file=sys.stderr,
        )

    # Wire up trimmer logging now that log_record exists
    if trimmer is not None:
        trimmer.log_fn = log_record

    @app.route("/v1/messages/count_tokens", methods=["POST"])
    def proxy_count_tokens():
        """Forward count_tokens to upstream, applying compaction first."""
        body = request.get_json(force=True)
        session = _get_session(body)
        sid = session["id"]
        ps = session["page_store"]

        # Apply the same compaction so the count reflects reality
        if compact and ps is not None:
            messages = body.get("messages", [])
            stats = compact_messages(
                messages,
                age_threshold=age_threshold,
                min_size=min_size,
                page_store=ps,
            )
            if stats.evicted_count > 0:
                print(
                    f"  [{sid}] [count_tokens] compacted {stats.evicted_count} results "
                    f"before counting",
                    file=sys.stderr,
                )

        headers = dict(request.headers)
        for h in ["Host", "Content-Length", "Transfer-Encoding"]:
            headers.pop(h, None)

        query = "/v1/messages/count_tokens"
        if request.query_string:
            query += "?" + request.query_string.decode("utf-8")

        count_body_bytes = len(json.dumps(body).encode("utf-8"))
        count_tools = len(body.get("tools", []))
        count_messages = len(body.get("messages", []))

        try:
            resp = client.post(query, json=body, headers=headers)
            # Parse and log the token count Anthropic returns
            input_tokens = None
            try:
                count_data = json.loads(resp.content)
                input_tokens = count_data.get("input_tokens")
            except Exception:
                pass
            # Stash for comparison with outgoing /v1/messages
            session["last_count"] = {
                "input_tokens": input_tokens,
                "body_bytes": count_body_bytes,
                "tools": count_tools,
                "messages": count_messages,
            }
            log_record({
                "type": "count_tokens",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session": sid,
                "input_tokens": input_tokens,
                "body_bytes": count_body_bytes,
                "tools": count_tools,
                "messages": count_messages,
                "status_code": resp.status_code,
            })
            print(
                f"  [{sid}] COUNT: {input_tokens} tokens"
                f" ({count_body_bytes / 1024:.0f}KB,"
                f" {count_messages} msgs, {count_tools} tools)",
                file=sys.stderr,
            )
            return (
                resp.content,
                resp.status_code,
                strip_response_headers(resp.headers),
            )
        except Exception as e:
            print(f"  [{sid}] [count_tokens] error: {e}", file=sys.stderr)
            return Response(
                json.dumps({"error": str(e)}),
                status=502,
                content_type="application/json",
            )

    @app.route("/v1/messages", methods=["POST"])
    def proxy_messages():
        """Proxy the messages endpoint with full logging."""
        request_time = datetime.now(timezone.utc)
        body = request.get_json(force=True)
        session = _get_session(body)
        sid = session["id"]
        ts = session["token_state"]
        ps = session["page_store"]

        # Measure without storing full content
        system_metrics = measure_system_prompt(body)
        message_metrics = measure_messages(body)

        request_record = {
            "type": "request",
            "timestamp": request_time.isoformat(),
            "session": sid,
            "model": body.get("model", "unknown"),
            "max_tokens": body.get("max_tokens"),
            "stream": body.get("stream", False),
            "system": system_metrics,
            "messages": message_metrics,
            "total_request_bytes": len(
                json.dumps(body).encode("utf-8")
            ),
        }

        if body.get("system"):
            request_record["system_prompt_full"] = body["system"]

        # Store full messages for offline replay
        request_record["messages_full"] = body.get("messages", [])

        log_record(request_record)

        # On first request for this session, calibrate turn counter
        if not ts["calibrated"]:
            messages = body.get("messages", [])
            prior_turns = sum(1 for m in messages if m.get("role") == "assistant")
            ts["turn"] = prior_turns
            ts["calibrated"] = True

        ts["turn"] += 1

        # --- Token cap gate ---
        if token_cap > 0 and ts["blocked"]:
            last = ts["last_effective"]
            msg = (
                f"Token cap exceeded. Last effective context: "
                f"{last:,} tokens (cap: {token_cap:,}). "
                f"Session has been blocked to prevent billing. "
                f"Restart with a fresh context or raise --token-cap."
            )
            print(
                f"{_RED}  [{sid}] BLOCKED: {msg}{_RESET}",
                file=sys.stderr,
            )
            log_record({
                "type": "token_cap",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session": sid,
                "action": "blocked",
                "last_effective_tokens": last,
                "token_cap": token_cap,
                "turn": ts["turn"],
            })
            return Response(
                json.dumps({
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": msg,
                    },
                }),
                status=429,
                content_type="application/json",
            )

        # --- System prompt trimming (if enabled) ---
        if trim and trimmer is not None:
            trim_result = trimmer.trim(body)
            if trim_result.total_bytes_saved > 0:
                print(
                    f"  [{sid}] Trimmed: {trim_result.tools.stubbed_tools} stubs, "
                    f"{trim_result.skills.duplicates_removed} skill dupes, "
                    f"saved {trim_result.total_bytes_saved:,} bytes",
                    file=sys.stderr,
                )

        # --- Temperature override (if configured) ---
        # Extended thinking requires temperature=1; skip override when thinking is active.
        thinking_enabled = bool(body.get("thinking"))
        if app.config.get("temperature_override") is not None and not thinking_enabled:
            body["temperature"] = app.config["temperature_override"]
        elif app.config.get("temperature_override") is not None and thinking_enabled:
            print(
                f"  [{sid}] [proxy] Skipping temperature override "
                f"(thinking enabled: {body.get('thinking')})",
                file=sys.stderr,
            )

        # --- System status injection (always active) ---
        # Give the model context pressure awareness and system identification.
        # Replaced each turn (not appended) so the model sees current state.
        inject_system_status(body, ts, token_cap, request_time,
                              block_store=session.get("block_store"))

        # --- Block labeling (conversation memory management) ---
        bs = session["block_store"]
        if compact and bs is not None:
            messages = body.get("messages", [])
            bs.label_messages(messages, ts["turn"])

        # --- Cleanup tag processing (Phase 2) ---
        if compact and bs is not None:
            messages = body.get("messages", [])
            cleanup_stats = process_cleanup_tags(messages, bs,
                                                  session.get("page_store"))
            if cleanup_stats:
                print(
                    f"  [{sid}] CLEANUP: {cleanup_stats}",
                    file=sys.stderr,
                )
            # Apply block status to messages (replace dropped/summarized)
            apply_stats = bs.apply_to_messages(messages)
            applied = sum(apply_stats.values())
            if applied:
                print(
                    f"  [{sid}] BLOCKS: {apply_stats}",
                    file=sys.stderr,
                )

        # --- Phantom tool handling (always active) ---
        observe_only = session["observe_only"]
        if ps is not None:
            # Inject results for phantom calls from the previous turn
            pending = session["phantom_pending"]
            if pending:
                messages = body.get("messages", [])
                inject_phantom_results(messages, pending, ps, observe_only)
                session["phantom_pending"] = []
                released = [
                    c for c in pending
                    if c.name in ("qunqay", "memory_release")
                ]
                faulted = [
                    c for c in pending
                    if c.name in ("yuyay", "recall", "memory_fault")
                ]
                if released:
                    paths = []
                    for c in released:
                        paths.extend(c.input.get("paths", []))
                    print(
                        f"  [{sid}] RELEASE: model released {len(paths)} path(s)",
                        file=sys.stderr,
                    )
                if faulted:
                    handles = []
                    for c in faulted:
                        handles.extend(c.input.get("handles",
                                       c.input.get("paths", [])))
                    print(
                        f"  [{sid}] RECALL: restored {len(handles)} tensor(s) "
                        f"from cache",
                        file=sys.stderr,
                    )

            # Inject phantom tools into the tools array
            observe_only = inject_tools(body)
            session["observe_only"] = observe_only

        # --- Context paging (if enabled) ---
        if compact and ps is not None:
            messages = body.get("messages", [])

            # Detect page faults BEFORE compaction
            faults = ps.detect_faults(messages)
            if faults:
                for fault in faults:
                    ago = time.monotonic() - fault.original_eviction.evicted_at
                    if ago < 60:
                        age_str = f"{ago:.0f}s ago"
                    else:
                        age_str = f"{ago / 60:.1f}min ago"
                    print(
                        f"  [{sid}] PAGE FAULT: {fault.tool_name} "
                        f"re-requested (evicted {age_str}, "
                        f"{fault.original_eviction.original_size:,} bytes)",
                        file=sys.stderr,
                    )
                log_record({
                    "type": "page_faults",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session": sid,
                    "count": len(faults),
                    "faults": [
                        {
                            "tool_name": f.tool_name,
                            "original_turn": f.original_eviction.turn_index,
                            "original_size": f.original_eviction.original_size,
                        }
                        for f in faults
                    ],
                    "cumulative_faults": len(ps.faults),
                    "unique_evictions": ps.unique_evictions,
                    "gc_count": ps.gc_count,
                    "fault_rate": ps.fault_rate,
                })

            # Compact
            stats = compact_messages(
                messages,
                age_threshold=age_threshold,
                min_size=min_size,
                page_store=ps,
            )
            if stats.evicted_count > 0:
                post_metrics = measure_messages(body)
                compaction_record = {
                    "type": "compaction",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session": sid,
                    "evicted": stats.evicted_count,
                    "total_tool_results": stats.total_tool_results,
                    "bytes_before": stats.bytes_before,
                    "bytes_after": stats.bytes_after,
                    "bytes_saved": stats.bytes_saved,
                    "reduction_pct": round(stats.reduction_pct, 1),
                    "skipped_small": stats.skipped_small,
                    "skipped_recent": stats.skipped_recent,
                    "skipped_error": stats.skipped_error,
                    "unique_evictions": ps.unique_evictions,
                    "gc_count": ps.gc_count,
                    "eviction_bytes_saved": ps.eviction_bytes_saved,
                    "gc_bytes_saved": ps.gc_bytes_saved,
                    "cumulative_faults": len(ps.faults),
                    "fault_rate": round(ps.fault_rate * 100, 1),
                    "messages_bytes_before": message_metrics[
                        "messages_total_bytes"
                    ],
                    "messages_bytes_after": post_metrics[
                        "messages_total_bytes"
                    ],
                }
                log_record(compaction_record)

                # Dashboard status line (pre-forward, bytes only)
                before_kb = message_metrics["messages_total_bytes"] / 1024
                after_kb = post_metrics["messages_total_bytes"] / 1024
                saved_pct = (
                    (1 - after_kb / before_kb) * 100
                    if before_kb > 0
                    else 0
                )
                pin_str = f" pin {len(ps._pinned)}" if ps._pinned else ""
                print(
                    f"  [{sid}] [{before_kb:.0f}KB \u2192 {after_kb:.0f}KB]"
                    f" ({saved_pct:.0f}% saved) "
                    f"ev {ps.unique_evictions} gc {ps.gc_count}"
                    f"{pin_str} | "
                    f"faults {len(ps.faults)}/{ps.unique_evictions}",
                    file=sys.stderr,
                )
            else:
                # No eviction this turn — still show working set size
                ws_kb = message_metrics["messages_total_bytes"] / 1024
                print(
                    f"  [{sid}] [{ws_kb:.0f}KB]",
                    file=sys.stderr,
                )

            # Conversation compression — DISABLED by default.
            # Third-party summarization loses model-relevant context;
            # the model should manage its own memory via qunqay/compact.
            # To re-enable, set PICHAY_CONV_COMPACT=1.
            conv_stats = None
            if os.environ.get("PICHAY_CONV_COMPACT") == "1":
                conv_stats = compact_conversation(messages, preserve_recent=12)
            if conv_stats and conv_stats.messages_compressed > 0:
                log_record({
                    "type": "conversation_compaction",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session": sid,
                    "messages_compressed": conv_stats.messages_compressed,
                    "chars_saved": conv_stats.chars_saved,
                })
                print(
                    f"  [{sid}] CONV: {conv_stats.messages_compressed} msgs compressed, "
                    f"{conv_stats.chars_saved:,} chars saved",
                    file=sys.stderr,
                )

        # Forward to Anthropic — measure the final outgoing payload
        outgoing_bytes = len(json.dumps(body).encode("utf-8"))
        outgoing_tools = len(body.get("tools", []))
        outgoing_messages = len(body.get("messages", []))
        log_record({
            "type": "outgoing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session": sid,
            "turn": ts["turn"],
            "outgoing_bytes": outgoing_bytes,
            "outgoing_tools": outgoing_tools,
            "outgoing_messages": outgoing_messages,
            "max_tokens": body.get("max_tokens"),
            "model": body.get("model", "unknown"),
        })
        # Compare with last count_tokens to detect divergence
        last_count = session.get("last_count")
        divergence_pct = None
        if last_count and last_count["body_bytes"] > 0:
            divergence_pct = (
                (outgoing_bytes - last_count["body_bytes"])
                / last_count["body_bytes"]
                * 100
            )

        if divergence_pct is not None and divergence_pct > 10:
            print(
                f"{_RED}  [{sid}] DIVERGENCE WARNING: outgoing "
                f"{outgoing_bytes / 1024:.0f}KB vs count_tokens "
                f"{last_count['body_bytes'] / 1024:.0f}KB "
                f"(+{divergence_pct:.0f}%) — gateway is injecting "
                f"{(outgoing_bytes - last_count['body_bytes']) / 1024:.0f}KB"
                f"{_RESET}",
                file=sys.stderr,
            )
            print(
                f"{_RED}  [{sid}]   count: {last_count['messages']} msgs, "
                f"{last_count['tools']} tools → outgoing: "
                f"{outgoing_messages} msgs, {outgoing_tools} tools"
                f"{_RESET}",
                file=sys.stderr,
            )
        else:
            print(
                f"  [{sid}] OUTGOING: {outgoing_bytes / 1024:.0f}KB"
                f" ({outgoing_messages} msgs, {outgoing_tools} tools)",
                file=sys.stderr,
            )

        headers = dict(request.headers)
        for h in ["Host", "Content-Length", "Transfer-Encoding"]:
            headers.pop(h, None)

        upstream_path = "/v1/messages"
        if request.query_string:
            upstream_path += "?" + request.query_string.decode("utf-8")

        if body.get("stream", False):
            return _proxy_streaming(
                body, headers, request_time, upstream_path, session,
            )
        else:
            return _proxy_direct(
                body, headers, request_time, upstream_path, session,
            )

    def _proxy_direct(body, headers, request_time, upstream_path, session):
        sid = session["id"]
        try:
            resp = client.post(
                upstream_path, json=body, headers=headers,
            )
            response_time = datetime.now(timezone.utc)
            try:
                resp_body = resp.json()
                usage = resp_body.get("usage", {})
                log_record({
                    "type": "response",
                    "timestamp": response_time.isoformat(),
                    "session": sid,
                    "duration_ms": int(
                        (response_time - request_time).total_seconds() * 1000
                    ),
                    "status_code": resp.status_code,
                    "usage": usage,
                    "stop_reason": resp_body.get("stop_reason"),
                })
                _check_token_cap(usage, session)
                _display_turn_status(usage, session)
            except Exception:
                log_record({
                    "type": "response_error",
                    "timestamp": response_time.isoformat(),
                    "session": sid,
                    "status_code": resp.status_code,
                })
            return Response(
                resp.content,
                status=resp.status_code,
                headers=strip_response_headers(resp.headers),
            )
        except Exception as e:
            log_record({
                "type": "proxy_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session": sid,
                "error": str(e),
            })
            return Response(
                json.dumps({"error": str(e)}),
                status=502,
                content_type="application/json",
            )

    def _proxy_streaming(body, headers, request_time, upstream_path, session):
        sid = session["id"]
        ps = session["page_store"]
        observe_only = session["observe_only"]
        try:
            resp = client.send(
                client.build_request(
                    "POST", upstream_path, json=body, headers=headers,
                ),
                stream=True,
            )
            # Log non-200 responses immediately (error bodies are not SSE)
            if resp.status_code != 200:
                error_body = resp.read().decode("utf-8", errors="replace")
                log_record({
                    "type": "api_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session": sid,
                    "status_code": resp.status_code,
                    "error_body": error_body[:2000],
                })
                print(
                    f"  [{sid}] API ERROR {resp.status_code}: "
                    f"{error_body[:200]}",
                    file=sys.stderr,
                )
                resp.close()
                return Response(
                    error_body,
                    status=resp.status_code,
                    headers=strip_response_headers(resp.headers),
                    content_type=resp.headers.get(
                        "content-type", "application/json"
                    ),
                )
            response_headers = strip_response_headers(resp.headers)

            def generate():
                first_byte_time = None
                chunks_collected = []
                phantom_calls: list[PhantomCall] = []
                continuation_needed: list[bool] = []
                try:
                    # Always filter stream for phantom tool calls
                    for chunk in filtered_stream(
                        resp.iter_bytes(),
                        chunks_collected,
                        phantom_calls,
                        observe_only=observe_only,
                        block_store=session["block_store"],
                        page_store=session.get("page_store"),
                        session_id=sid,
                        continuation_needed=continuation_needed,
                    ):
                        if first_byte_time is None:
                            first_byte_time = datetime.now(timezone.utc)
                        yield chunk
                finally:
                    resp.close()
                    response_time = datetime.now(timezone.utc)
                    full_response = b"".join(chunks_collected)

                    # Handle phantom calls — gateway mode.
                    # The proxy executes phantom tools and auto-continues
                    # so the model gets results without a turn boundary.
                    if phantom_calls:
                        for pc in phantom_calls:
                            _handle_phantom_call(pc, ps,
                                                block_store=session.get("block_store"))
                            print(
                                f"  [{sid}] PHANTOM{'(observe)' if observe_only else ''}: "
                                f"{pc.name}({pc.input})",
                                file=sys.stderr,
                            )

                    # Auto-continue: if filtered_stream suppressed the stop
                    # events, we need to execute phantom tools and send the
                    # results back so the model can continue generating.
                    if continuation_needed and phantom_calls and not observe_only:
                        try:
                            yield from _phantom_continuation(
                                body, phantom_calls, ps,
                                session.get("block_store"),
                                headers, upstream_path, client,
                                chunks_collected, sid,
                            )
                        except Exception as e:
                            print(
                                f"  [{sid}] CONTINUATION ERROR: {e}",
                                file=sys.stderr,
                            )
                            # Fallback: yield stop events so stream is valid
                            stop_delta = json.dumps({
                                "type": "message_delta",
                                "delta": {"stop_reason": "end_turn"},
                                "usage": {},
                            })
                            stop_msg = json.dumps({"type": "message_stop"})
                            yield f"data: {stop_delta}\n\n".encode()
                            yield f"data: {stop_msg}\n\n".encode()
                            yield b"data: [DONE]\n\n"

                    usage = {}
                    try:
                        text = full_response.decode("utf-8", errors="replace")
                        # Merge usage from both message_start (input tokens)
                        # and message_delta (output tokens). Don't break early.
                        for line in text.split("\n"):
                            if not line.startswith("data: "):
                                continue
                            if "message_start" in line and "usage" in line:
                                event_data = json.loads(line[6:])
                                msg = event_data.get("message", {})
                                usage.update(msg.get("usage", {}))
                            elif "message_delta" in line and "usage" in line:
                                event_data = json.loads(line[6:])
                                usage.update(event_data.get("usage", {}))
                    except Exception:
                        pass

                    log_record({
                        "type": "response_stream",
                        "timestamp": response_time.isoformat(),
                        "session": sid,
                        "duration_ms": int(
                            (response_time - request_time).total_seconds()
                            * 1000
                        ),
                        "first_byte_ms": int(
                            (first_byte_time - request_time).total_seconds()
                            * 1000
                        )
                        if first_byte_time
                        else None,
                        "status_code": resp.status_code,
                        "response_bytes": len(full_response),
                        "usage": usage,
                    })
                    _check_token_cap(usage, session)
                    _display_turn_status(usage, session)

            return Response(
                generate(),
                status=resp.status_code,
                headers=response_headers,
                content_type=response_headers.get(
                    "content-type", "text/event-stream"
                ),
            )
        except Exception as e:
            log_record({
                "type": "proxy_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session": sid,
                "error": str(e),
            })
            return Response(
                json.dumps({"error": str(e)}),
                status=502,
                content_type="application/json",
            )

    @app.route("/health")
    def health():
        parts = []
        if compact:
            parts.append("compact")
        if trim:
            parts.append("trim")
        mode = "+".join(parts) if parts else "observe"

        result = {
            "status": "ok",
            "mode": mode,
            "log_file": str(log_file),
            "active_sessions": len(_sessions),
        }
        result["sessions"] = {}
        for sid, s in _sessions.items():
            sess_info = s["page_store"].summary() if s["page_store"] else {}
            if s.get("block_store"):
                sess_info["blocks"] = s["block_store"].summary()
            result["sessions"][sid] = sess_info
        if trim and trimmer is not None:
            result["trimmer"] = trimmer.summary()
        return result

    print(f"Logging to: {log_file}", file=sys.stderr)
    return app


def main():
    parser = argparse.ArgumentParser(
        description="Logging proxy for Claude API (with optional context paging)"
    )
    parser.add_argument(
        "--port", type=int, default=0,
        help="Port to listen on (0 = random free port, default: 0)",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"),
        help="Directory for proxy logs (default: logs/)",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Enable context paging: evict stale tool results",
    )
    parser.add_argument(
        "--trim", action="store_true",
        help="Enable system prompt trimming: tool stubs, skill dedup, static tracking",
    )
    parser.add_argument(
        "--age-threshold", type=int, default=4,
        help="Evict tool results older than N user-turns (default: 4)",
    )
    parser.add_argument(
        "--min-size", type=int, default=500,
        help="Don't evict results smaller than N bytes (default: 500)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override temperature on all requests (e.g., 0 for deterministic)",
    )
    parser.add_argument(
        "--upstream", type=str, default=DEFAULT_API_BASE,
        help=f"Upstream API base URL (default: {DEFAULT_API_BASE}). "
             "Any Anthropic-compatible endpoint: OpenRouter, Kimi, etc.",
    )
    parser.add_argument(
        "--token-cap", type=int, default=0,
        help="Hard cap on effective input tokens. Blocks requests after "
             "exceeding this. Warning at 80%%. 0 = no cap (default: 0). "
             "Set to 200000 for subscription billing threshold.",
    )
    args = parser.parse_args()

    app = create_app(
        args.log_dir,
        compact=args.compact,
        trim=args.trim,
        age_threshold=args.age_threshold,
        min_size=args.min_size,
        upstream=args.upstream,
        token_cap=args.token_cap,
    )
    if args.temperature is not None:
        app.config["temperature_override"] = args.temperature

    port = args.port if args.port != 0 else find_free_port()
    parts = []
    if args.compact:
        parts.append("COMPACT")
    if args.trim:
        parts.append("TRIM")
    mode = "+".join(parts) if parts else "OBSERVE"
    print(
        f"Proxy [{mode}] listening on http://localhost:{port}",
        file=sys.stderr,
    )
    print(
        f"Use: ANTHROPIC_BASE_URL=http://localhost:{port} claude",
        file=sys.stderr,
    )
    app.run(host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    main()
