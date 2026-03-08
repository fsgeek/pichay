from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from pichay.gateway import create_app
from pichay.telemetry import Telemetry


class FakeStreamResponse:
    def __init__(self, *, status_code: int = 200, chunks: list[bytes] | None = None, error_after_chunks: int | None = None):
        self.status_code = status_code
        self._chunks = chunks or []
        self._error_after_chunks = error_after_chunks
        self.headers = httpx.Headers({"content-type": "text/event-stream"})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_bytes(self):
        for i, chunk in enumerate(self._chunks):
            if self._error_after_chunks is not None and i >= self._error_after_chunks:
                raise httpx.ReadError("stream interrupted")
            yield chunk

    def read(self) -> bytes:
        return b"".join(self._chunks)


class FakeClient:
    def __init__(self, *, stream_response: FakeStreamResponse | None = None, stream_error: Exception | None = None):
        self._stream_response = stream_response
        self._stream_error = stream_error

    def stream(self, method, path, json=None, headers=None):
        if self._stream_error is not None:
            raise self._stream_error
        assert self._stream_response is not None
        return self._stream_response

    def post(self, path, json=None, headers=None):
        return httpx.Response(200, json={"ok": True})

    def close(self):
        return None


BASE_PAYLOAD = {
    "model": "claude-test",
    "max_tokens": 64,
    "stream": True,
    "messages": [{"role": "user", "content": "hello"}],
}


def _events(app):
    return app.state.telemetry.recent_events()


def test_malformed_stream_chunk_emits_invariant_violation(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        process_session_id="proc_test",
    )
    app.state.clients["anthropic"] = FakeClient(
        stream_response=FakeStreamResponse(
            chunks=[
                b'data: {"type":"message_start"}\n\n',
                b"data: {not-json}\n\n",
                b"data: [DONE]\n\n",
            ]
        )
    )

    client = TestClient(app)
    with client.stream("POST", "/v1/messages", json=BASE_PAYLOAD) as resp:
        body = b"".join(resp.iter_bytes())
    assert b"{not-json}" in body

    events = _events(app)
    assert any(e.get("type") == "invariant_violation" and e.get("kind") == "malformed_stream_chunk" for e in events)
    assert any(e.get("type") == "request_metrics" for e in events)


def test_midstream_disconnect_emits_stream_error_and_terminal_metrics(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        process_session_id="proc_test",
    )
    app.state.clients["anthropic"] = FakeClient(
        stream_response=FakeStreamResponse(
            chunks=[
                b'data: {"type":"message_start"}\n\n',
                b'data: {"type":"content_block_delta"}\n\n',
            ],
            error_after_chunks=1,
        )
    )

    client = TestClient(app)
    with pytest.raises(Exception):
        with client.stream("POST", "/v1/messages", json=BASE_PAYLOAD) as resp:
            _ = list(resp.iter_bytes())

    events = _events(app)
    assert any(e.get("type") == "stream_error" for e in events)
    terminal = [e for e in events if e.get("type") == "request_metrics"]
    assert terminal
    assert terminal[-1].get("status") == 599


def test_upstream_timeout_emits_stream_error_and_terminal_metrics(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        process_session_id="proc_test",
    )
    app.state.clients["anthropic"] = FakeClient(
        stream_error=httpx.ReadTimeout("timeout"),
    )

    client = TestClient(app)
    with pytest.raises(Exception):
        with client.stream("POST", "/v1/messages", json=BASE_PAYLOAD) as resp:
            _ = list(resp.iter_bytes())

    events = _events(app)
    assert any(e.get("type") == "stream_error" for e in events)
    terminal = [e for e in events if e.get("type") == "request_metrics"]
    assert terminal
    assert terminal[-1].get("status") == 599


# ── Resilience: upstream 4xx in streaming mode ─────────────────────


def test_upstream_4xx_streaming_returns_error_body(tmp_path: Path):
    """When upstream returns 4xx on a streaming request, gateway should
    read the body and return it rather than trying to iterate chunks."""
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    error_body = b'{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}'
    app.state.clients["anthropic"] = FakeClient(
        stream_response=FakeStreamResponse(
            status_code=529,
            chunks=[error_body],
        )
    )

    client = TestClient(app)
    with client.stream("POST", "/v1/messages", json=BASE_PAYLOAD) as resp:
        body = b"".join(resp.iter_bytes())

    assert b"overloaded" in body
    events = _events(app)
    metrics = [e for e in events if e.get("type") == "request_metrics"]
    assert metrics
    assert metrics[-1]["status"] == 529


# ── Resilience: non-streaming provider error returns 502 ───────────


class _ErrorClient:
    def post(self, path, json=None, headers=None):
        raise httpx.ConnectError("connection refused")

    def stream(self, method, path, json=None, headers=None):
        raise httpx.ConnectError("connection refused")

    def close(self):
        pass


def test_non_streaming_provider_error_returns_502(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    app.state.clients["anthropic"] = _ErrorClient()

    client = TestClient(app)
    resp = client.post("/v1/messages", json={
        "model": "claude-test", "max_tokens": 64, "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 502

    events = _events(app)
    assert any(e.get("type") == "provider_error" for e in events)


# ── Resilience: connection refused on streaming path ───────────────


def test_connection_refused_streaming_emits_stream_error(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    app.state.clients["anthropic"] = FakeClient(
        stream_error=httpx.ConnectError("connection refused"),
    )

    client = TestClient(app)
    with pytest.raises(Exception):
        with client.stream("POST", "/v1/messages", json=BASE_PAYLOAD) as resp:
            _ = list(resp.iter_bytes())

    events = _events(app)
    assert any(e.get("type") == "stream_error" for e in events)
    metrics = [e for e in events if e.get("type") == "request_metrics"]
    assert metrics
    assert metrics[-1]["status"] == 599


# ── Observability: health endpoint shape ───────────────────────────


def test_health_endpoint_returns_expected_shape(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_health",
    )
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["process_session_id"] == "proc_health"
    assert "anthropic" in body["providers"]
    assert "openai" in body["providers"]
    assert "log_path" in body


# ── Observability: metrics endpoint returns prometheus text ────────


def test_metrics_endpoint_returns_prometheus_format(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert b"pichay_requests_total" in resp.content


# ── Observability: sessions API accumulates across requests ────────


def test_sessions_api_accumulates(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_sess",
    )
    app.state.clients["anthropic"] = FakeClient(
        stream_response=FakeStreamResponse(
            chunks=[b'data: {"type":"message_start"}\n\n', b"data: [DONE]\n\n"],
        )
    )

    client = TestClient(app)
    for _ in range(2):
        with client.stream("POST", "/v1/messages", json=BASE_PAYLOAD) as resp:
            _ = b"".join(resp.iter_bytes())

    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    # Sessions are keyed by per-conversation fingerprint, not process ID
    assert len(sessions) >= 1
    session = next(iter(sessions.values()))
    assert session["request_count"] == 2
    assert session["incoming_bytes"] > 0


# ── Observability: events API with valid window ────────────────────


def test_events_api_with_valid_window(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    app.state.telemetry.emit("test_event", data="hello")

    client = TestClient(app)
    resp = client.get("/api/events?window=1h")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert any(e.get("type") == "test_event" for e in events)

    resp = client.get("/api/events")
    assert resp.status_code == 200
    assert len(resp.json()["events"]) >= 1


# ── Observability: telemetry emits outgoing>incoming invariant ─────


def test_telemetry_emits_outgoing_larger_invariant(tmp_path: Path):
    t = Telemetry(log_path=tmp_path / "test.jsonl", hydration_window_seconds=3600)
    t.record_request(
        request_id="r1", session_id="s1", provider="anthropic",
        model="x", status=200, incoming_bytes=100, outgoing_bytes=200,
        latency_ms=10.0, streaming=False, duplication_score=0.0,
    )
    events = t.recent_events()
    violations = [e for e in events if e.get("type") == "invariant_violation"
                  and e.get("kind") == "outgoing_larger_than_incoming"]
    assert violations
    assert violations[0]["incoming_bytes"] == 100
    assert violations[0]["outgoing_bytes"] == 200


# ── Observability: telemetry log persists and hydrates ─────────────


def test_telemetry_hydrates_from_log(tmp_path: Path):
    log_path = tmp_path / "hydrate.jsonl"
    t1 = Telemetry(log_path=log_path, hydration_window_seconds=3600)
    t1.emit("test_persist", data="rehydrated")

    t2 = Telemetry(log_path=log_path, hydration_window_seconds=3600)
    events = t2.recent_events()
    assert any(e.get("type") == "test_persist" for e in events)


# ── Observability: request_metrics includes expected fields ────────


def test_request_metrics_event_has_all_fields(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    app.state.clients["anthropic"] = FakeClient(
        stream_response=FakeStreamResponse(
            chunks=[b'data: {"type":"message_start"}\n\n', b"data: [DONE]\n\n"],
        )
    )

    client = TestClient(app)
    with client.stream("POST", "/v1/messages", json=BASE_PAYLOAD) as resp:
        _ = b"".join(resp.iter_bytes())

    events = _events(app)
    metrics = [e for e in events if e.get("type") == "request_metrics"]
    assert metrics
    m = metrics[-1]
    required_fields = {
        "request_id", "session_id", "provider", "model", "status",
        "incoming_bytes", "outgoing_bytes", "shrink_ratio",
        "duplication_score", "latency_ms", "streaming", "timestamp",
    }
    assert required_fields.issubset(set(m.keys())), f"Missing: {required_fields - set(m.keys())}"
    assert m["provider"] == "anthropic"
    assert m["model"] == "claude-test"
    assert m["streaming"] is True
    assert isinstance(m["shrink_ratio"], float)
