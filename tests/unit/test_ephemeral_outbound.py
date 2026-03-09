from __future__ import annotations

import copy

import httpx

from pichay.gateway import create_app

STATUS_ANCHOR_TEXT = "[pichay-live-status]"


def _make_app(tmp_path):
    return create_app(
        log_dir=tmp_path,
        anthropic_upstream="http://anthropic.test",
        openai_upstream="http://openai.test",
        hydration_window_seconds=24 * 3600,
        enable_paging=False,
        enable_trim=False,
        min_evict_size=500,
        process_session_id="ephemeral-test",
    )


class RecordingClient:
    def __init__(self, usage_tokens: int = 2048):
        self.calls: list[dict] = []
        self.usage_tokens = usage_tokens

    def post(self, path, json=None, headers=None):
        self.calls.append({"path": path, "json": copy.deepcopy(json)})
        return httpx.Response(200, json={"completion": {}, "usage": {"input_tokens": self.usage_tokens}})

    def close(self):
        return None


def _get_handle_provider_request(app):
    handler = None
    for route in app.router.routes:
        methods = getattr(route, "methods", set()) or set()
        if getattr(route, "path", None) == "/v1/messages" and "POST" in methods:
            handler = route.endpoint
            break
    if handler is None:
        raise AssertionError("/v1/messages route not found")

    for cell in handler.__closure__ or []:
        candidate = cell.cell_contents
        if callable(candidate) and getattr(candidate, "__name__", "") == "_handle_provider_request":
            return candidate
    raise AssertionError("_handle_provider_request handle not found")


def _extract_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "\n".join(texts)
    return ""


def test_status_injection_uses_ephemeral_copy(tmp_path):
    app = _make_app(tmp_path)
    recording_client = RecordingClient()
    app.state.clients["anthropic"] = recording_client
    handle_request = _get_handle_provider_request(app)

    user_message = {"role": "user", "content": "First turn message"}
    payload = {
        "model": "claude-test",
        "max_tokens": 64,
        "stream": False,
        "messages": [user_message],
    }

    session = app.state.sessions.get(payload)
    session.token_state["last_effective"] = 1500

    request_payload = dict(payload)
    request_payload["_headers"] = {}
    resp = handle_request("anthropic", "messages", request_payload, "")
    assert resp.status_code == 200

    assert recording_client.calls
    physical_last = session.message_store.messages[-1]
    assert STATUS_ANCHOR_TEXT not in _extract_text(physical_last)

    outbound_last = recording_client.calls[-1]["json"]["messages"][-1]
    assert STATUS_ANCHOR_TEXT in _extract_text(outbound_last)


def test_prior_messages_remain_clean_on_second_turn(tmp_path):
    app = _make_app(tmp_path)
    recording_client = RecordingClient()
    app.state.clients["anthropic"] = recording_client
    handle_request = _get_handle_provider_request(app)

    first_user = {"role": "user", "content": "Turn 1 question"}
    first_payload = {
        "model": "claude-test",
        "max_tokens": 64,
        "stream": False,
        "messages": [first_user],
    }

    session = app.state.sessions.get(first_payload)
    session.token_state["last_effective"] = 2000

    first_request_payload = dict(first_payload)
    first_request_payload["_headers"] = {}
    resp1 = handle_request("anthropic", "messages", first_request_payload, "")
    assert resp1.status_code == 200

    session.token_state["last_effective"] = 2500
    second_payload = {
        "model": "claude-test",
        "max_tokens": 64,
        "stream": False,
        "messages": [
            {"role": first_user["role"], "content": first_user["content"]},
            {"role": "assistant", "content": "Turn 1 answer"},
            {"role": "user", "content": "Turn 2 follow-up"},
        ],
    }

    second_request_payload = dict(second_payload)
    second_request_payload["_headers"] = {}
    resp2 = handle_request("anthropic", "messages", second_request_payload, "")
    assert resp2.status_code == 200

    assert len(recording_client.calls) >= 2
    second_call_body = recording_client.calls[-1]["json"]
    first_user_outbound = second_call_body["messages"][0]
    assert STATUS_ANCHOR_TEXT not in _extract_text(first_user_outbound)

    final_user_outbound = second_call_body["messages"][-1]
    assert STATUS_ANCHOR_TEXT in _extract_text(final_user_outbound)
