from __future__ import annotations

from datetime import datetime
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from pichay.gateway import create_app
from pichay.telemetry import Telemetry


REQUEST_METRICS_REQUIRED = {
    "type",
    "timestamp",
    "request_id",
    "session_id",
    "provider",
    "model",
    "status",
    "incoming_bytes",
    "outgoing_bytes",
    "shrink_ratio",
    "duplication_score",
    "latency_ms",
    "streaming",
    "input_tokens",
    "cache_read_tokens",
    "cache_create_tokens",
    "effective_tokens",
    "miss_penalty_tokens_est",
    "size_saved_tokens_est",
    "net_token_value_est",
}

POLICY_CONFLICT_REQUIRED = {
    "type",
    "timestamp",
    "winner_stage",
    "loser_stage",
    "loser_action",
    "target_id",
    "target_bytes",
    "duplication_score",
    "resolution_reason",
}


class _FakeNonStreamClient:
    def __init__(self, response: httpx.Response):
        self._response = response

    def post(self, path, json=None, headers=None):
        return self._response

    def close(self):
        pass


class _FakeStreamResponse:
    def __init__(self, *, chunks: list[bytes], status_code: int = 200):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = httpx.Headers({"content-type": "text/event-stream"})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_bytes(self):
        yield from self._chunks

    def read(self) -> bytes:
        return b"".join(self._chunks)


class _FakeStreamClient:
    def __init__(self, response: _FakeStreamResponse):
        self._response = response

    def stream(self, method, path, json=None, headers=None):
        return self._response

    def close(self):
        pass


def _req_event(app) -> dict:
    events = app.state.telemetry.recent_events()
    req = [e for e in events if e.get("type") == "request_metrics"]
    assert req, "expected request_metrics event"
    return req[-1]


def test_request_metrics_contract_anthropic_nonstream(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        process_session_id="proc_contract",
    )
    app.state.clients["anthropic"] = _FakeNonStreamClient(
        httpx.Response(200, json={"id": "ok"})
    )

    client = TestClient(app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-test",
            "max_tokens": 16,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200

    ev = _req_event(app)
    assert REQUEST_METRICS_REQUIRED.issubset(ev.keys())
    datetime.fromisoformat(ev["timestamp"])
    assert ev["provider"] == "anthropic"
    assert ev["streaming"] is False
    assert isinstance(ev["shrink_ratio"], float)


def test_request_metrics_contract_openai_stream(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        process_session_id="proc_contract",
    )
    app.state.clients["openai"] = _FakeStreamClient(
        _FakeStreamResponse(
            chunks=[
                b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"hi"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    )

    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-test",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    ) as resp:
        _ = b"".join(resp.iter_bytes())

    ev = _req_event(app)
    assert REQUEST_METRICS_REQUIRED.issubset(ev.keys())
    assert ev["provider"] == "openai"
    assert ev["streaming"] is True


def test_policy_conflict_contract_fields(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=True,
        enable_trim=True,
        min_evict_size=100,
        process_session_id="proc_contract",
    )
    app.state.clients["anthropic"] = _FakeNonStreamClient(
        httpx.Response(200, json={"id": "ok"})
    )

    big = "dup" * 300
    client = TestClient(app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-test",
            "max_tokens": 16,
            "stream": False,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": big, "pichay_phantom_protected": True}]},
                {"role": "assistant", "content": [{"type": "text", "text": big}]},
            ],
        },
    )
    assert resp.status_code == 200

    events = app.state.telemetry.recent_events()
    conflicts = [e for e in events if e.get("type") == "policy_conflict_resolved"]
    assert conflicts
    c = conflicts[-1]
    assert POLICY_CONFLICT_REQUIRED.issubset(c.keys())
    assert c["winner_stage"] == "phantom"


def test_telemetry_invariant_contract_direct(tmp_path: Path):
    t = Telemetry(log_path=tmp_path / "events.jsonl", hydration_window_seconds=24 * 3600)
    t.record_request(
        request_id="r1",
        session_id="s1",
        provider="anthropic",
        model="m",
        status=200,
        incoming_bytes=10,
        outgoing_bytes=20,
        latency_ms=1.2,
        streaming=False,
        duplication_score=0.0,
    )
    events = t.recent_events()
    inv = [e for e in events if e.get("type") == "anomaly"]
    assert inv
    e = inv[-1]
    assert e["kind"] == "outgoing_growth_suspicious"
    assert e["incoming_bytes"] == 10
    assert e["outgoing_bytes"] == 20


def test_smoke_both_provider_routes_emit_metrics(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        process_session_id="proc_contract",
    )
    app.state.clients["anthropic"] = _FakeNonStreamClient(httpx.Response(200, json={"id": "a"}))
    app.state.clients["openai"] = _FakeStreamClient(
        _FakeStreamResponse(chunks=[b'data: {"id":"o"}\n\n', b"data: [DONE]\n\n"])
    )

    client = TestClient(app)
    r1 = client.post(
        "/v1/messages",
        json={
            "model": "claude-test",
            "max_tokens": 16,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r1.status_code == 200

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-test",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    ) as r2:
        _ = b"".join(r2.iter_bytes())

    req = [e for e in app.state.telemetry.recent_events() if e.get("type") == "request_metrics"]
    providers = {e["provider"] for e in req}
    assert providers == {"anthropic", "openai"}


def test_cost_summary_endpoint_shape(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        process_session_id="proc_contract",
    )
    app.state.clients["anthropic"] = _FakeNonStreamClient(httpx.Response(200, json={
        "id": "a",
        "usage": {
            "input_tokens": 1000,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 900,
        },
    }))
    client = TestClient(app)
    _ = client.post(
        "/v1/messages",
        json={
            "model": "claude-test",
            "max_tokens": 16,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    resp = client.get("/api/cost?window=1h")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "requests",
        "avg_cache_read_pct",
        "zero_cache_read_requests",
        "avg_effective_tokens",
        "avg_miss_penalty_tokens_est",
        "avg_size_saved_tokens_est",
        "avg_net_token_value_est",
    ):
        assert key in body
