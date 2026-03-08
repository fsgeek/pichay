"""Pichay gateway — cache-aware multi-provider proxy for LLM context management.

Primary entry point. Supersedes deprecated/proxy.py (Flask).

Architecture:
    Claude Code → Gateway (FastAPI) → Provider (Anthropic, OpenAI)

The gateway intercepts, transforms, and forwards API requests. Features:
- Per-conversation session management with stable fingerprinting
- Static system prompt injection (cache-friendly, no prefix mutation)
- Dynamic status anchor (end-of-messages, after cache breakpoints)
- Token cap enforcement with cache hit rate tracking
- Policy pipeline: phantom protection → paging → trim
- Multi-provider support via adapter pattern
- Prometheus metrics + live HTML dashboard
"""

from __future__ import annotations

import argparse
import hashlib
from contextlib import asynccontextmanager
import json
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from pichay.blocks import BlockStore
from pichay.core.models import CanonicalRequest
from pichay.core.pipeline import Pipeline
from pichay.core.policy import PolicyConfig, collect_blocks
from pichay.core.utils import parse_duration
from pichay.launcher import LaunchSpec, launch
from pichay.message_ops import (
    check_inbound_for_injected_tags,
    get_system_prompt,
    inject_system_status,
    process_cleanup_tags,
    PICHAY_STATUS_MARKER,
)
from pichay.pager import PageStore
from pichay.providers import adapters
from pichay.telemetry import Telemetry


# ANSI for stderr status lines
_DIM = "\033[2m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _session_id(body: dict) -> str:
    """Derive a stable session fingerprint from the first user message.

    Different conversations have different first messages, so this
    is unique per conversation and stable across turns.
    """
    messages = body.get("messages", [])
    first = json.dumps(messages[0], sort_keys=True) if messages else ""
    return hashlib.sha256(first.encode()).hexdigest()[:8]


def _duplication_score(req: CanonicalRequest) -> float:
    texts = [b.text for b in collect_blocks(req) if b.text]
    if not texts:
        return 0.0
    unique = len(set(texts))
    return max(0.0, float(len(texts) - unique) / float(len(texts)))


def _copy_headers(headers: httpx.Headers) -> dict[str, str]:
    dropped = {
        "content-length",
        "transfer-encoding",
        "connection",
        "content-encoding",
    }
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in dropped:
            continue
        out[k] = v
    return out


def _inspect_sse_chunk(
    chunk: bytes,
    *,
    buffer: bytearray,
    emit_event,
    request_id: str,
    session_id: str,
    provider: str,
    usage_accumulator: dict[str, Any] | None = None,
) -> None:
    """Best-effort SSE validation for telemetry; never mutates payload."""
    buffer.extend(chunk)
    while b"\n\n" in buffer:
        raw_event, rest = buffer.split(b"\n\n", 1)
        buffer[:] = rest
        if not raw_event:
            continue
        try:
            text = raw_event.decode("utf-8")
        except UnicodeDecodeError as e:
            emit_event(
                "invariant_violation",
                kind="malformed_stream_chunk",
                request_id=request_id,
                session_id=session_id,
                provider=provider,
                error=f"invalid_utf8:{e}",
            )
            continue

        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                evt = json.loads(payload)
                if usage_accumulator is not None and isinstance(evt, dict):
                    etype = evt.get("type")
                    if etype == "message_start":
                        msg = evt.get("message", {})
                        if isinstance(msg, dict):
                            usage = msg.get("usage", {})
                            if isinstance(usage, dict):
                                usage_accumulator.update(usage)
                    elif etype == "message_delta":
                        usage = evt.get("usage", {})
                        if isinstance(usage, dict):
                            usage_accumulator.update(usage)
            except json.JSONDecodeError as e:
                emit_event(
                    "invariant_violation",
                    kind="malformed_stream_chunk",
                    request_id=request_id,
                    session_id=session_id,
                    provider=provider,
                    error=f"invalid_json:{e}",
                )


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pichay Gateway Dashboard</title>
  <style>
    :root {
      --bg0: #0b1220;
      --bg1: #172034;
      --panel: #101a2e;
      --text: #e6edf8;
      --muted: #9cb0d0;
      --ok: #18c29c;
      --warn: #f6c84c;
      --err: #ef6b73;
      --accent: #57a3ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: radial-gradient(1200px 700px at 10% -20%, #24365f 0%, transparent 55%),
                  radial-gradient(1000px 600px at 90% 120%, #1b4f66 0%, transparent 55%),
                  linear-gradient(160deg, var(--bg0), var(--bg1));
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      min-height: 100vh;
    }
    header {
      padding: 18px 24px;
      border-bottom: 1px solid #2b3a57;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }
    .title { font-size: 1.4rem; letter-spacing: 0.3px; }
    .meta { color: var(--muted); font-size: 0.95rem; }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 14px;
      padding: 16px;
    }
    .panel {
      background: linear-gradient(180deg, #11203a 0%, #0f1a2e 100%);
      border: 1px solid #2b3a57;
      border-radius: 12px;
      padding: 12px 14px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }
    .kpi { grid-column: span 3; }
    .wide { grid-column: span 8; }
    .mid { grid-column: span 4; }
    .full { grid-column: span 12; }
    @media (max-width: 980px) {
      .kpi, .wide, .mid, .full { grid-column: span 12; }
    }
    h2 { margin: 0 0 10px 0; font-size: 1rem; color: #cfe0ff; }
    .v { font-size: 1.65rem; color: var(--accent); }
    .sub { color: var(--muted); font-size: 0.9rem; }
    .status-ok { color: var(--ok); }
    .status-warn { color: var(--warn); }
    .status-err { color: var(--err); }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #263754; }
    th { color: #c8d9f8; font-weight: 600; }
    .events {
      max-height: 320px;
      overflow: auto;
      font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      background: #0b1529;
      border: 1px solid #24334f;
      padding: 10px;
      border-radius: 8px;
    }
    .spark {
      width: 100%;
      height: 140px;
      background: #0b1529;
      border: 1px solid #24334f;
      border-radius: 8px;
      position: relative;
      overflow: hidden;
    }
    .spark svg { width: 100%; height: 100%; display: block; }
  </style>
</head>
<body>
  <header>
    <div class="title">Pichay Gateway Dashboard</div>
    <div class="meta" id="meta">loading...</div>
  </header>
  <main class="grid">
    <section class="panel kpi"><h2>Requests</h2><div class="v" id="kpi-req">0</div><div class="sub">total observed</div></section>
    <section class="panel kpi"><h2>Shrink Ratio</h2><div class="v" id="kpi-shrink">1.00</div><div class="sub">avg outgoing/incoming</div></section>
    <section class="panel kpi"><h2>Violations</h2><div class="v status-warn" id="kpi-viol">0</div><div class="sub">invariant alerts</div></section>
    <section class="panel kpi"><h2>Stream Errors</h2><div class="v status-err" id="kpi-se">0</div><div class="sub">terminal stream failures</div></section>

    <section class="panel wide">
      <h2>Shrink Ratio Trend (recent requests)</h2>
      <div class="spark" id="spark-wrap"><svg id="spark"></svg></div>
    </section>
    <section class="panel mid">
      <h2>Health</h2>
      <div class="sub" id="health">loading...</div>
    </section>

    <section class="panel full">
      <h2>Session Totals</h2>
      <table>
        <thead><tr><th>Session</th><th>Requests</th><th>Incoming Bytes</th><th>Outgoing Bytes</th></tr></thead>
        <tbody id="sessions"></tbody>
      </table>
    </section>

    <section class="panel full">
      <h2>Recent Events</h2>
      <div class="events" id="events"></div>
    </section>
  </main>
  <script>
    function fmt(n) { return Number(n || 0).toLocaleString(); }
    function toFixed(n, d=2) { return Number(n || 0).toFixed(d); }

    function renderSpark(values) {
      const svg = document.getElementById('spark');
      const w = svg.clientWidth || 600;
      const h = svg.clientHeight || 140;
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      if (!values.length) { svg.innerHTML = ''; return; }
      const maxV = Math.max(...values, 1.2);
      const minV = Math.min(...values, 0.5);
      const pad = 10;
      const xstep = (w - 2 * pad) / Math.max(values.length - 1, 1);
      const y = v => h - pad - ((v - minV) / Math.max(maxV - minV, 0.01)) * (h - 2 * pad);
      let d = '';
      values.forEach((v, i) => { d += `${i===0?'M':'L'} ${pad + i*xstep} ${y(v)} `; });
      svg.innerHTML = `
        <path d="${d}" stroke="#57a3ff" stroke-width="2.2" fill="none"></path>
        <line x1="0" y1="${y(1.0)}" x2="${w}" y2="${y(1.0)}" stroke="#355a89" stroke-dasharray="4 4"></line>
      `;
    }

    async function load() {
      const [health, sessions, eventsResp] = await Promise.all([
        fetch('/health').then(r => r.json()),
        fetch('/api/sessions').then(r => r.json()),
        fetch('/api/events?window=24h').then(r => r.json())
      ]);

      document.getElementById('meta').textContent =
        `process ${health.process_session_id} • providers: ${health.providers.join(', ')}`;
      document.getElementById('health').textContent =
        `${health.status} | log ${health.log_path}`;

      const ev = eventsResp.events || [];
      const req = ev.filter(e => e.type === 'request_metrics');
      const viol = ev.filter(e => e.type === 'invariant_violation');
      const se = ev.filter(e => e.type === 'stream_error');
      const avgShrink = req.length ? req.reduce((a, r) => a + Number(r.shrink_ratio || 1), 0) / req.length : 1.0;

      document.getElementById('kpi-req').textContent = fmt(req.length);
      document.getElementById('kpi-shrink').textContent = toFixed(avgShrink, 3);
      document.getElementById('kpi-viol').textContent = fmt(viol.length);
      document.getElementById('kpi-se').textContent = fmt(se.length);

      renderSpark(req.slice(-80).map(r => Number(r.shrink_ratio || 1)));

      const tbody = document.getElementById('sessions');
      tbody.innerHTML = '';
      const entries = Object.entries(sessions.sessions || {});
      entries.sort((a,b) => (b[1].request_count||0) - (a[1].request_count||0));
      for (const [sid, s] of entries) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${sid}</td><td>${fmt(s.request_count)}</td><td>${fmt(s.incoming_bytes)}</td><td>${fmt(s.outgoing_bytes)}</td>`;
        tbody.appendChild(tr);
      }

      const evBox = document.getElementById('events');
      evBox.textContent = ev.slice(-120).map(x => JSON.stringify(x)).join('\\n');
      evBox.scrollTop = evBox.scrollHeight;
    }

    load().catch(err => {
      document.getElementById('meta').textContent = `dashboard error: ${err}`;
    });
    setInterval(() => load().catch(() => {}), 2500);
  </script>
</body>
</html>
"""


# ── Session state ────────────────────────────────────────────────────────

class Session:
    """Per-conversation state. Isolated by first-message fingerprint."""

    def __init__(self, sid: str, log_dir: Path):
        self.id = sid
        self.token_state = {
            "last_effective": 0,
            "blocked": False,
            "turn": 0,
        }
        self.page_store = PageStore(
            log_path=log_dir / f"pages_{sid}.jsonl",
        )
        self._block_checkpoint = log_dir / f"blocks_{sid}.json"
        if self._block_checkpoint.is_file():
            self.block_store = BlockStore.from_checkpoint(self._block_checkpoint)
            print(
                f"  {_DIM}[{sid}] restored {self.block_store.block_count} blocks{_RESET}",
                file=sys.stderr,
            )
        else:
            self.block_store = BlockStore()

    def track_usage(self, usage: dict) -> None:
        """Update token state from API response usage."""
        effective = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        if effective > 0:
            self.token_state["last_effective"] = effective

    def increment_turn(self) -> None:
        self.token_state["turn"] += 1


class SessionStore:
    """Maps conversation fingerprints to session state."""

    def __init__(self, log_dir: Path):
        self._sessions: dict[str, Session] = {}
        self._log_dir = log_dir

    def get(self, body: dict) -> Session:
        sid = _session_id(body)
        if sid not in self._sessions:
            self._sessions[sid] = Session(sid, self._log_dir)
            print(
                f"  {_DIM}[{sid}] new session{_RESET}",
                file=sys.stderr,
            )
        return self._sessions[sid]

    def all(self) -> dict[str, Session]:
        return dict(self._sessions)


# ── Gateway factory ──────────────────────────────────────────────────────

def create_app(
    *,
    log_dir: Path,
    anthropic_upstream: str = "https://api.anthropic.com",
    openai_upstream: str = "https://api.openai.com",
    hydration_window_seconds: int = 86400,
    enable_paging: bool = True,
    enable_trim: bool = True,
    min_evict_size: int = 500,
    token_cap: int = 0,
    process_session_id: str | None = None,
) -> FastAPI:
    clients: dict[str, httpx.Client] = {}

    if process_session_id is None:
        process_session_id = datetime.now(timezone.utc).strftime("proc_%Y%m%d_%H%M%S")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            for c in clients.values():
                c.close()

    app = FastAPI(title="Pichay Gateway", version="0.3.0", lifespan=lifespan)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"gateway_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    telemetry = Telemetry(log_path=log_path, hydration_window_seconds=hydration_window_seconds)
    app.state.telemetry = telemetry
    app.state.process_session_id = process_session_id

    sessions = SessionStore(log_dir)
    app.state.sessions = sessions

    _token_warning_threshold = int(token_cap * 0.80) if token_cap > 0 else 0

    cfg = PolicyConfig(
        enable_paging=enable_paging,
        enable_trim=enable_trim,
        min_evict_size=min_evict_size,
    )

    def emit_event(event_type: str, **fields: Any) -> None:
        telemetry.emit(
            event_type,
            process_session_id=process_session_id,
            **fields,
        )

    pipeline = Pipeline(cfg=cfg, emit_event=emit_event)
    app.state.pipeline = pipeline

    app.state.adapters = adapters()
    clients.update({
        "anthropic": httpx.Client(base_url=anthropic_upstream, timeout=httpx.Timeout(300.0, connect=30.0)),
        "openai": httpx.Client(base_url=openai_upstream, timeout=httpx.Timeout(300.0, connect=30.0)),
    })
    app.state.clients = clients

    # ── Endpoints ────────────────────────────────────────────────────

    @app.get("/health")
    def health() -> dict[str, Any]:
        all_sessions = sessions.all()
        session_summaries = {}
        for sid, s in all_sessions.items():
            ps = s.page_store
            session_summaries[sid] = {
                "turn": s.token_state["turn"],
                "last_effective_tokens": s.token_state["last_effective"],
                "evictions": ps.unique_evictions,
                "gc": ps.gc_count,
                "faults": len(ps.faults),
            }
        return {
            "status": "ok",
            "process_session_id": process_session_id,
            "providers": ["anthropic", "openai"],
            "log_path": str(log_path),
            "token_cap": token_cap,
            "sessions": session_summaries,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=telemetry.get_metrics(), media_type="text/plain; version=0.0.4")

    @app.get("/api/sessions")
    def api_sessions() -> dict[str, Any]:
        return {
            "process_session_id": process_session_id,
            "sessions": telemetry.session_summary(),
        }

    @app.get("/api/events")
    def api_events(window: str | None = None) -> dict[str, Any]:
        window_seconds = None
        if window:
            try:
                window_seconds = parse_duration(window)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        return {"events": telemetry.recent_events(window_seconds)}

    @app.get("/dashboard")
    def dashboard() -> HTMLResponse:
        return HTMLResponse(_dashboard_html())

    # ── Pre/post processing ──────────────────────────────────────────

    def _preprocess(payload: dict, session: Session) -> dict:
        """Apply gateway transformations before pipeline and forwarding.

        Operates on the raw Anthropic-format payload (before normalization)
        because system prompt injection needs access to the system field
        directly.
        """
        session.increment_turn()
        ts = session.token_state
        request_time = datetime.now(timezone.utc)

        # System status: static system prompt + dynamic anchor
        inject_system_status(
            payload, ts, token_cap, request_time,
            block_store=session.block_store,
        )

        # Block labeling (with injection-safe validation)
        messages = payload.get("messages", [])
        session.block_store.label_messages(messages, ts["turn"])

        return payload

    def _check_token_cap(usage: dict, session: Session) -> None:
        """Track usage and enforce token cap."""
        session.track_usage(usage)
        if token_cap <= 0:
            return

        effective = session.token_state["last_effective"]
        pct = effective / token_cap * 100
        sid = session.id

        if effective > token_cap:
            session.token_state["blocked"] = True
            print(
                f"{_RED}  [{sid}] TOKEN CAP EXCEEDED: {effective:,} / {token_cap:,} "
                f"({pct:.0f}%) — next request will be blocked{_RESET}",
                file=sys.stderr,
            )

    def _display_turn_status(usage: dict, session: Session) -> None:
        """Post-response status line with cache hit rate."""
        sid = session.id
        ts = session.token_state
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

        # Cache hit rate
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)
        cache_total = cache_read + cache_create
        cache_str = ""
        if cache_total > 0:
            cache_pct = cache_read / cache_total * 100
            cache_str = f" | cache {cache_pct:.0f}%"

        ps = session.page_store
        ev_str = ""
        if ps.unique_evictions > 0 or ps.gc_count > 0:
            pin_str = f" pin {len(ps._pinned)}" if ps._pinned else ""
            ev_str = (
                f" | ev {ps.unique_evictions} gc {ps.gc_count}{pin_str}"
                f" | faults {len(ps.faults)}/{ps.unique_evictions}"
            )

        print(
            f"  [{sid}] [Turn {ts['turn']}] "
            f"{effective:,} tok{cap_str}{cache_str}{ev_str}",
            file=sys.stderr,
        )

    # ── Request handling ─────────────────────────────────────────────

    def _handle_provider_request(
        provider: str,
        endpoint: str,
        payload: dict[str, Any],
        query_string: str = "",
    ):
        payload = dict(payload)
        forwarded_headers = payload.pop("_headers", {})

        # Session management
        session = sessions.get(payload)

        # Token cap enforcement
        if token_cap > 0 and session.token_state.get("blocked"):
            raise HTTPException(
                status_code=429,
                detail=f"Token cap exceeded ({session.token_state['last_effective']:,}/{token_cap:,})",
            )

        # Reject inbound cleanup tag injection
        # tag_error = check_inbound_for_injected_tags(payload)
        # if tag_error:
        #     print(f"  {_RED}[{session.id}] {tag_error}{_RESET}", file=sys.stderr)
        #     raise HTTPException(status_code=400, detail=tag_error)

        # Process cleanup tags from assistant messages (model is TCB)
        if endpoint == "messages":
            messages = payload.get("messages", [])
            cleanup_stats = process_cleanup_tags(
                messages, session.block_store, session.page_store,
            )
            if cleanup_stats:
                print(
                    f"  {_DIM}[{session.id}] cleanup: {cleanup_stats}{_RESET}",
                    file=sys.stderr,
                )
                # Checkpoint block state after cleanup mutations
                saved = session.block_store.checkpoint(session._block_checkpoint)
                print(
                    f"  {_DIM}[{session.id}] checkpointed {saved} blocks{_RESET}",
                    file=sys.stderr,
                )

        # Pre-process: system status, block labeling
        if endpoint == "messages":
            payload = _preprocess(payload, session)

        adapter = app.state.adapters[provider]
        client: httpx.Client = app.state.clients[provider]

        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        incoming_bytes = len(json.dumps(payload, default=str).encode("utf-8"))
        session_id = session.id

        req = adapter.normalize_request(payload)
        req = pipeline.run(req)
        duplication_score = _duplication_score(req)
        outgoing_body = adapter.denormalize_request(req)
        if endpoint == "count_tokens":
            outgoing_body.pop("stream", None)
            outgoing_body.pop("max_tokens", None)
        outgoing_bytes = len(json.dumps(outgoing_body, default=str).encode("utf-8"))

        upstream_path = adapter.upstream_path(req, endpoint=endpoint)
        if query_string:
            upstream_path = f"{upstream_path}?{query_string}"
        headers = {
            k: v for k, v in forwarded_headers.items()
            if k.lower() not in {"host", "content-length", "transfer-encoding"}
        }

        if req.stream:
            def generate():
                status_code = 599
                bytes_out = 0
                chunk_count = 0
                stream_error = False
                sse_buffer = bytearray()
                usage: dict[str, Any] = {}
                try:
                    with client.stream("POST", upstream_path, json=outgoing_body, headers=headers) as resp:
                        status_code = resp.status_code
                        if status_code >= 400:
                            body = resp.read()
                            bytes_out = len(body)
                            yield body
                        else:
                            for chunk in resp.iter_bytes():
                                bytes_out += len(chunk)
                                chunk_count += 1
                                _inspect_sse_chunk(
                                    chunk,
                                    buffer=sse_buffer,
                                    emit_event=emit_event,
                                    request_id=request_id,
                                    session_id=session_id,
                                    provider=provider,
                                    usage_accumulator=usage,
                                )
                                yield chunk
                except Exception as e:
                    stream_error = True
                    emit_event(
                        "stream_error",
                        request_id=request_id,
                        session_id=session_id,
                        provider=provider,
                        error=str(e),
                    )
                    raise
                finally:
                    if stream_error:
                        status_code = 599
                    # Post-response: track usage, display status
                    if usage:
                        _check_token_cap(usage, session)
                        _display_turn_status(usage, session)
                    emit_event(
                        "response_observed",
                        request_id=request_id,
                        session_id=session_id,
                        provider=provider,
                        endpoint=endpoint,
                        status=status_code,
                        response_bytes=bytes_out,
                        chunk_count=chunk_count,
                        usage=usage,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000
                    telemetry.record_request(
                        request_id=request_id,
                        session_id=session_id,
                        provider=provider,
                        model=req.model,
                        status=status_code,
                        incoming_bytes=incoming_bytes,
                        outgoing_bytes=outgoing_bytes,
                        latency_ms=latency_ms,
                        streaming=True,
                        duplication_score=duplication_score,
                        usage=usage,
                    )
            return StreamingResponse(generate(), media_type="text/event-stream")

        # Non-streaming
        try:
            resp = client.post(upstream_path, json=outgoing_body, headers=headers)
        except httpx.HTTPError as e:
            emit_event(
                "provider_error",
                request_id=request_id,
                session_id=session_id,
                provider=provider,
                error=str(e),
            )
            raise HTTPException(status_code=502, detail=str(e))

        usage: dict[str, Any] = {}
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                u = parsed.get("usage", {})
                if isinstance(u, dict):
                    usage = u
        except Exception:
            usage = {}

        if usage:
            _check_token_cap(usage, session)
            _display_turn_status(usage, session)

        emit_event(
            "response_observed",
            request_id=request_id,
            session_id=session_id,
            provider=provider,
            endpoint=endpoint,
            status=resp.status_code,
            response_bytes=len(resp.content),
            chunk_count=0,
            usage=usage,
        )

        latency_ms = (time.perf_counter() - started) * 1000
        telemetry.record_request(
            request_id=request_id,
            session_id=session_id,
            provider=provider,
            model=req.model,
            status=resp.status_code,
            incoming_bytes=incoming_bytes,
            outgoing_bytes=outgoing_bytes,
            latency_ms=latency_ms,
            streaming=False,
            duplication_score=duplication_score,
            usage=usage,
        )
        return Response(content=resp.content, status_code=resp.status_code, headers=_copy_headers(resp.headers))

    def _payload_with_headers(req: Request, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["_headers"] = dict(req.headers)
        return payload

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        payload = await request.json()
        return _handle_provider_request(
            "anthropic",
            "messages",
            _payload_with_headers(request, payload),
            query_string=request.url.query,
        )

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        payload = await request.json()
        payload = _payload_with_headers(request, payload)
        payload["stream"] = False
        return _handle_provider_request(
            "anthropic",
            "count_tokens",
            payload,
            query_string=request.url.query,
        )

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request):
        payload = await request.json()
        return _handle_provider_request(
            "openai",
            "chat_completions",
            _payload_with_headers(request, payload),
            query_string=request.url.query,
        )

    return app


def _run_server_in_thread(app: FastAPI, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config=config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait up to 10s for bind.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if server.started:
            return server, t
        time.sleep(0.05)
    raise RuntimeError("gateway server did not start")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pichay multi-provider gateway")
    parser.add_argument("--claude", action="store_true", help="Launch Claude CLI against gateway")
    parser.add_argument("--codex", action="store_true", help="Launch Codex CLI against gateway")
    parser.add_argument("--gemini", action="store_true", help="Gemini mode (stub in v1)")
    parser.add_argument("--no-launch", action="store_true", help="Run gateway as persistent service")
    parser.add_argument("--port", type=int, default=0, help="Gateway port (0=random)")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Telemetry log directory")
    parser.add_argument("--hydration-window", default="24h", help="Dashboard hydration window (e.g. 24h, 7d)")
    parser.add_argument("--anthropic-upstream", default="https://api.anthropic.com")
    parser.add_argument("--openai-upstream", default="https://api.openai.com")
    parser.add_argument("--token-cap", type=int, default=0, help="Token cap (0=unlimited)")
    parser.add_argument("--disable-paging", action="store_true")
    parser.add_argument("--disable-trim", action="store_true")
    parser.add_argument("--min-evict-size", type=int, default=500)
    parser.add_argument("cli_args", nargs=argparse.REMAINDER, help="Args passed to launched CLI after '--'")
    args = parser.parse_args()

    selected_modes = [m for m, enabled in (("claude", args.claude), ("codex", args.codex), ("gemini", args.gemini)) if enabled]
    if not args.no_launch and len(selected_modes) != 1:
        parser.error("choose exactly one launch target: --claude | --codex | --gemini, or pass --no-launch")

    try:
        hydration_seconds = parse_duration(args.hydration_window)
    except ValueError as e:
        parser.error(str(e))

    port = args.port if args.port != 0 else find_free_port()

    app = create_app(
        log_dir=args.log_dir,
        anthropic_upstream=args.anthropic_upstream,
        openai_upstream=args.openai_upstream,
        hydration_window_seconds=hydration_seconds,
        enable_paging=not args.disable_paging,
        enable_trim=not args.disable_trim,
        min_evict_size=args.min_evict_size,
        token_cap=args.token_cap,
    )

    if args.no_launch:
        print(f"Gateway listening on http://127.0.0.1:{port}", file=sys.stderr)
        uvicorn.run(app, host="127.0.0.1", port=port)
        return

    mode = selected_modes[0]
    if mode == "gemini":
        print("Gemini adapter is not enabled in v1. Use --claude or --codex.", file=sys.stderr)
        raise SystemExit(2)

    server, thread = _run_server_in_thread(app, port)
    print(f"Gateway listening on http://127.0.0.1:{port}", file=sys.stderr)

    extra = args.cli_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    rc = 1
    try:
        rc = launch(LaunchSpec(mode=mode, port=port, extra_args=extra))
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
