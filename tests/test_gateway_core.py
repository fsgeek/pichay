from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from pichay.core.models import CanonicalMessage, CanonicalRequest
from pichay.core.pipeline import Pipeline
from pichay.core.policy import PolicyConfig, apply_action, paging_stage, trim_stage
from pichay.core.utils import content_bytes, parse_duration
from pichay.gateway import _copy_headers, _duplication_score, create_app
from pichay.providers.anthropic import AnthropicAdapter
from pichay.providers.openai import OpenAIAdapter


def test_parse_duration():
    assert parse_duration("24h") == 24 * 3600
    assert parse_duration("15m") == 15 * 60
    assert parse_duration("7d") == 7 * 86400


def test_policy_precedence_phantom_blocks_trim_and_paging():
    events: list[dict] = []

    def emit(event_type: str, **fields):
        events.append({"type": event_type, **fields})

    req = CanonicalRequest(
        provider="anthropic",
        model="x",
        max_tokens=100,
        stream=False,
        messages=[
            CanonicalMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "duplicate-content" * 50,
                        "pichay_phantom_protected": True,
                    }
                ],
            ),
            CanonicalMessage(
                role="assistant",
                content=[
                    {
                        "type": "text",
                        "text": "duplicate-content" * 50,
                    }
                ],
            ),
        ],
    )

    p = Pipeline(PolicyConfig(enable_paging=True, enable_trim=True, min_evict_size=10), emit)
    p.run(req)

    conflict_events = [e for e in events if e["type"] == "policy_conflict_resolved"]
    assert conflict_events
    assert any(e["winner_stage"] == "phantom" for e in conflict_events)


def test_anthropic_adapter_roundtrip():
    adapter = AnthropicAdapter()
    payload = {
        "model": "claude-test",
        "max_tokens": 100,
        "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "system": "sys",
        "x_custom": 1,
    }
    req = adapter.normalize_request(payload)
    out = adapter.denormalize_request(req)
    assert out["model"] == payload["model"]
    assert out["x_custom"] == 1
    assert out["messages"][0]["content"][0]["text"] == "hello"


def test_openai_adapter_roundtrip():
    adapter = OpenAIAdapter()
    payload = {
        "model": "gpt-test",
        "max_tokens": 200,
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "response_format": {"type": "json_object"},
    }
    req = adapter.normalize_request(payload)
    out = adapter.denormalize_request(req)
    assert out["model"] == payload["model"]
    assert out["response_format"] == payload["response_format"]
    assert out["messages"][0]["content"] == "hello"


def test_dashboard_route_serves_html(tmp_path: Path):
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
    client = TestClient(app)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Pichay Gateway Dashboard" in resp.text


def test_api_events_invalid_window_returns_400(tmp_path: Path):
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
    client = TestClient(app)
    resp = client.get("/api/events?window=bogus")
    assert resp.status_code == 400


# ── parse_duration edge cases ──────────────────────────────────────


def test_parse_duration_seconds():
    assert parse_duration("30s") == 30


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("abc")
    with pytest.raises(ValueError):
        parse_duration("")
    with pytest.raises(ValueError):
        parse_duration("10x")


# ── _duplication_score ─────────────────────────────────────────────


def test_duplication_score_no_duplicates():
    req = CanonicalRequest(
        provider="anthropic", model="x", max_tokens=10, stream=False,
        messages=[CanonicalMessage(role="user", content=[{"type": "text", "text": "unique"}])],
    )
    assert _duplication_score(req) == 0.0


def test_duplication_score_all_duplicates():
    req = CanonicalRequest(
        provider="anthropic", model="x", max_tokens=10, stream=False,
        messages=[
            CanonicalMessage(role="user", content=[{"type": "text", "text": "dup"}]),
            CanonicalMessage(role="assistant", content=[{"type": "text", "text": "dup"}]),
        ],
    )
    # 2 blocks, 1 unique → score = (2-1)/2 = 0.5
    assert _duplication_score(req) == pytest.approx(0.5)


def test_duplication_score_empty_messages():
    req = CanonicalRequest(
        provider="anthropic", model="x", max_tokens=10, stream=False, messages=[],
    )
    assert _duplication_score(req) == 0.0


# ── _copy_headers ──────────────────────────────────────────────────


def test_copy_headers_drops_hop_by_hop():
    h = httpx.Headers({
        "content-type": "application/json",
        "content-length": "123",
        "transfer-encoding": "chunked",
        "connection": "keep-alive",
        "content-encoding": "gzip",
        "x-custom": "kept",
    })
    out = _copy_headers(h)
    assert "content-type" in out
    assert "x-custom" in out
    assert "content-length" not in out
    assert "transfer-encoding" not in out
    assert "connection" not in out
    assert "content-encoding" not in out


# ── content_bytes ──────────────────────────────────────────────────


def test_content_bytes_nested():
    assert content_bytes("hello") == 5
    assert content_bytes({"text": "abc"}) == 3
    assert content_bytes(["a", "b"]) == 2


# ── Pipeline actually shrinks payload ──────────────────────────────


def test_pipeline_paging_replaces_duplicate_blocks():
    events: list[dict] = []
    big_text = "x" * 600  # > min_evict_size

    req = CanonicalRequest(
        provider="anthropic", model="x", max_tokens=10, stream=False,
        messages=[
            CanonicalMessage(role="user", content=[{"type": "tool_result", "text": big_text}]),
            CanonicalMessage(role="user", content=[{"type": "tool_result", "text": big_text}]),
        ],
    )

    p = Pipeline(PolicyConfig(enable_paging=True, enable_trim=False, min_evict_size=100),
                 lambda t, **f: events.append({"type": t, **f}))
    result = p.run(req)

    applied = [e for e in events if e["type"] == "policy_action_applied"]
    assert applied
    # At least one block should now contain a paging tombstone.
    compacted = []
    for m in result.messages:
        for b in m.content:
            txt = b.get("text")
            ctt = b.get("content")
            if (isinstance(txt, str) and txt.startswith("[Paged out duplicate block:")) or (
                isinstance(ctt, str) and ctt.startswith("[Paged out duplicate block:")
            ):
                compacted.append(b)
    assert compacted


def test_pipeline_trim_replaces_exact_duplicate_within_role():
    events: list[dict] = []

    req = CanonicalRequest(
        provider="anthropic", model="x", max_tokens=10, stream=False,
        messages=[
            CanonicalMessage(role="user", content=[{"type": "text", "text": "same"}]),
            CanonicalMessage(role="user", content=[{"type": "text", "text": "same"}]),
        ],
    )

    p = Pipeline(PolicyConfig(enable_paging=False, enable_trim=True, min_evict_size=10),
                 lambda t, **f: events.append({"type": t, **f}))
    result = p.run(req)

    applied = [e for e in events if e["type"] == "policy_action_applied" and e.get("stage") == "trim"]
    assert applied


# ── apply_action edge cases ────────────────────────────────────────


def test_apply_action_out_of_bounds_returns_false():
    from pichay.core.models import PolicyAction

    req = CanonicalRequest(
        provider="anthropic", model="x", max_tokens=10, stream=False,
        messages=[CanonicalMessage(role="user", content=[{"type": "text", "text": "a"}])],
    )
    action = PolicyAction(
        stage="paging", action="evict", target_id="m99:b0",
        message_index=99, block_index=0, replacement_text="[gone]", bytes=1, duplication_score=1.0,
    )
    assert apply_action(req, action) is False


def test_apply_action_none_replacement_returns_false():
    from pichay.core.models import PolicyAction

    req = CanonicalRequest(
        provider="anthropic", model="x", max_tokens=10, stream=False,
        messages=[CanonicalMessage(role="user", content=[{"type": "text", "text": "a"}])],
    )
    action = PolicyAction(
        stage="paging", action="evict", target_id="m0:b0",
        message_index=0, block_index=0, replacement_text=None, bytes=1, duplication_score=1.0,
    )
    assert apply_action(req, action) is False


# ── paging_stage respects min_evict_size ───────────────────────────


def test_paging_stage_ignores_small_blocks():
    from pichay.core.models import PolicyContext

    req = CanonicalRequest(
        provider="anthropic", model="x", max_tokens=10, stream=False,
        messages=[
            CanonicalMessage(role="user", content=[{"type": "tool_result", "text": "tiny"}]),
            CanonicalMessage(role="user", content=[{"type": "tool_result", "text": "tiny"}]),
        ],
    )
    ctx = PolicyContext()
    actions = paging_stage(req, ctx, PolicyConfig(enable_paging=True, min_evict_size=9999))
    assert actions == []


# ── trim_stage skips non-duplicate blocks ──────────────────────────


def test_trim_stage_no_action_on_unique_content():
    from pichay.core.models import PolicyContext

    req = CanonicalRequest(
        provider="anthropic", model="x", max_tokens=10, stream=False,
        messages=[
            CanonicalMessage(role="user", content=[{"type": "text", "text": "alpha"}]),
            CanonicalMessage(role="user", content=[{"type": "text", "text": "beta"}]),
        ],
    )
    ctx = PolicyContext()
    actions = trim_stage(req, ctx, PolicyConfig(enable_trim=True))
    assert actions == []


# ── Non-streaming request path ─────────────────────────────────────


class _FakeNonStreamClient:
    def __init__(self, *, response: httpx.Response):
        self._response = response

    def post(self, path, json=None, headers=None):
        return self._response

    def close(self):
        pass


class _CaptureNonStreamClient(_FakeNonStreamClient):
    def __init__(self, *, response: httpx.Response):
        super().__init__(response=response)
        self.last_path = None
        self.last_json = None
        self.last_headers = None

    def post(self, path, json=None, headers=None):
        self.last_path = path
        self.last_json = json
        self.last_headers = headers
        return self._response


class _FakeStreamResponse:
    def __init__(self, *, status_code: int = 200, chunks: list[bytes] | None = None):
        self.status_code = status_code
        self._chunks = chunks or []
        self.headers = httpx.Headers({"content-type": "text/event-stream"})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_bytes(self):
        yield from self._chunks

    def read(self):
        return b"".join(self._chunks)


class _FakeStreamClient:
    def __init__(self, *, stream_response: _FakeStreamResponse):
        self._stream_response = stream_response

    def stream(self, method, path, json=None, headers=None):
        return self._stream_response

    def close(self):
        pass


def test_non_streaming_request_returns_upstream_body(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    upstream_body = {"id": "msg_123", "content": [{"type": "text", "text": "hi"}]}
    app.state.clients["anthropic"] = _FakeNonStreamClient(
        response=httpx.Response(200, json=upstream_body),
    )

    client = TestClient(app)
    resp = client.post("/v1/messages", json={
        "model": "claude-test", "max_tokens": 64, "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    assert resp.json()["id"] == "msg_123"

    events = app.state.telemetry.recent_events()
    metrics = [e for e in events if e.get("type") == "request_metrics"]
    assert metrics
    assert metrics[-1]["streaming"] is False
    observed = [e for e in events if e.get("type") == "response_observed"]
    assert observed
    assert observed[-1]["status"] == 200
    assert observed[-1]["endpoint"] == "messages"


# ── OpenAI route ───────────────────────────────────────────────────


def test_openai_chat_completions_route(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    app.state.clients["openai"] = _FakeStreamClient(
        stream_response=_FakeStreamResponse(
            chunks=[
                b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"hi"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    )

    client = TestClient(app)
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "gpt-4", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    }) as resp:
        body = b"".join(resp.iter_bytes())

    assert b"chatcmpl-1" in body
    events = app.state.telemetry.recent_events()
    metrics = [e for e in events if e.get("type") == "request_metrics"]
    assert metrics
    assert metrics[-1]["provider"] == "openai"
    observed = [e for e in events if e.get("type") == "response_observed"]
    assert observed
    assert observed[-1]["provider"] == "openai"
    assert observed[-1]["endpoint"] == "chat_completions"


# ── Count tokens endpoint ─────────────────────────────────────────


def test_count_tokens_endpoint(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    upstream_body = {"input_tokens": 42}
    cap = _CaptureNonStreamClient(
        response=httpx.Response(200, json=upstream_body),
    )
    app.state.clients["anthropic"] = cap

    client = TestClient(app)
    resp = client.post("/v1/messages/count_tokens", json={
        "model": "claude-test", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 42
    assert "stream" not in (cap.last_json or {})
    assert "max_tokens" not in (cap.last_json or {})

    events = app.state.telemetry.recent_events()
    metrics = [e for e in events if e.get("type") == "request_metrics"]
    assert metrics


def test_query_string_is_forwarded_to_upstream(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    cap = _CaptureNonStreamClient(response=httpx.Response(200, json={"ok": True}))
    app.state.clients["anthropic"] = cap

    client = TestClient(app)
    resp = client.post("/v1/messages/count_tokens?beta=true", json={
        "model": "claude-test", "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    assert cap.last_path == "/v1/messages/count_tokens?beta=true"


def test_internal_headers_not_forwarded_in_body(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False, enable_trim=False,
        min_evict_size=500, process_session_id="proc_test",
    )
    cap = _CaptureNonStreamClient(response=httpx.Response(200, json={"ok": True}))
    app.state.clients["anthropic"] = cap

    client = TestClient(app)
    resp = client.post("/v1/messages", json={
        "model": "claude-test", "max_tokens": 64, "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    assert "_headers" not in (cap.last_json or {})


def test_anthropic_model_override_applied(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        anthropic_model_override="claude-haiku-4-5",
        process_session_id="proc_test",
    )
    cap = _CaptureNonStreamClient(response=httpx.Response(200, json={"ok": True}))
    app.state.clients["anthropic"] = cap
    client = TestClient(app)
    resp = client.post("/v1/messages", json={
        "model": "claude-opus-4-6",
        "max_tokens": 64,
        "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    assert cap.last_json is not None
    assert cap.last_json.get("model") == "claude-haiku-4-5"


def test_openai_model_override_applied(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        openai_model_override="gpt-4o-mini",
        process_session_id="proc_test",
    )
    cap = _CaptureNonStreamClient(response=httpx.Response(200, json={"ok": True}))
    app.state.clients["openai"] = cap
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-4.1",
        "max_tokens": 64,
        "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    assert cap.last_json is not None
    assert cap.last_json.get("model") == "gpt-4o-mini"


def test_compacted_tool_result_preserves_anthropic_schema(tmp_path: Path):
    app = create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=True,
        enable_trim=False,
        min_evict_size=100,
        process_session_id="proc_test",
    )
    cap = _CaptureNonStreamClient(response=httpx.Response(200, json={"ok": True}))
    app.state.clients["anthropic"] = cap

    duplicated = "X" * 400
    payload = {
        "model": "claude-test",
        "max_tokens": 64,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a1", "content": duplicated},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a2", "content": duplicated},
                ],
            },
        ],
    }

    client = TestClient(app)
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200
    out = cap.last_json
    assert out is not None

    compacted = [
        b
        for m in out.get("messages", [])
        for b in m.get("content", [])
        if b.get("type") == "tool_result"
        and isinstance(b.get("content"), str)
        and b.get("content", "").startswith("[Paged out duplicate block:")
    ]
    assert compacted, "expected at least one compacted tool_result block"
    for block in compacted:
        assert "content" in block
        assert "text" not in block
        assert "pichay_compacted" not in block


# ── OpenAI adapter multi-block denormalize ─────────────────────────


def test_openai_adapter_multi_text_blocks_joined():
    adapter = OpenAIAdapter()
    req = CanonicalRequest(
        provider="openai", model="gpt-4", max_tokens=10, stream=False,
        messages=[CanonicalMessage(role="user", content=[
            {"type": "text", "text": "line1"},
            {"type": "text", "text": "line2"},
        ])],
    )
    out = adapter.denormalize_request(req)
    # Multiple text blocks should be joined with newline
    assert out["messages"][0]["content"] == "line1\nline2"


def test_openai_adapter_non_text_blocks_kept_as_list():
    adapter = OpenAIAdapter()
    req = CanonicalRequest(
        provider="openai", model="gpt-4", max_tokens=10, stream=False,
        messages=[CanonicalMessage(role="user", content=[
            {"type": "text", "text": "line1"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ])],
    )
    out = adapter.denormalize_request(req)
    assert isinstance(out["messages"][0]["content"], list)


# ── Anthropic adapter upstream_path ────────────────────────────────


def test_anthropic_adapter_upstream_path():
    adapter = AnthropicAdapter()
    req = CanonicalRequest(provider="anthropic", model="x", max_tokens=10, stream=False, messages=[])
    assert adapter.upstream_path(req, endpoint="messages") == "/v1/messages"
    assert adapter.upstream_path(req, endpoint="count_tokens") == "/v1/messages/count_tokens"
