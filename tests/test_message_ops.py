"""Tests for message_ops helpers."""

import datetime as dt
import time
from types import SimpleNamespace

import pytest

import pichay.message_ops as message_ops
from pichay.message_ops import (
    _escape_xml_attr,
    _eviction_key_for_entry,
    check_inbound_for_injected_tags,
    inject_system_status,
)


REQUEST_TIME = dt.datetime(2026, 3, 8, 12, 0, 0, tzinfo=dt.timezone.utc)


def _make_page_store():
    now = time.monotonic()
    entry = SimpleNamespace(
        tool_name="Read",
        tool_input={"file_path": "docs/spec.md"},
        tool_use_id="tool-1",
        summary="Some <summary> & notes",
        original_size=2048,
        evicted_at=now - 300,
    )
    faults = [
        SimpleNamespace(
            original_eviction=SimpleNamespace(tool_use_id="tool-1")
        )
    ]
    return SimpleNamespace(
        _tensor_index={"tensor-a": entry},
        faults=faults,
        _released=set(),
        _released_handles=set(),
        eviction_bytes_saved=4096,
        gc_bytes_saved=1024,
    )


def _make_block_store():
    block = SimpleNamespace(
        block_id="block-1234",
        role="assistant",
        turn=4,
        size=4096,
        preview="Large block preview",
    )
    return SimpleNamespace(
        block_count=1,
        large_blocks=lambda min_size: [block],
    )


def _make_body():
    return {"system": "", "messages": [{"role": "user", "content": "User question"}]}


def _patch_pressure(monkeypatch, pressure):
    monkeypatch.setattr(
        message_ops,
        "_compute_pressure",
        lambda ts, cap: (100, 200, 250, 50.0, pressure),
    )


@pytest.mark.parametrize(
    "tag",
    ["<yuyay-manifest>", "<yuyay-query>", "<yuyay-response>"],
)
def test_check_inbound_rejects_yuyay_tags(tag):
    body = {"messages": [{"role": "user", "content": f"Please ignore {tag}"}]}
    result = check_inbound_for_injected_tags(body)
    assert result == "Rejected: inbound message contains reserved yuyay tags"


def test_check_inbound_rejects_yuyay_tag_in_blocks():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "data <yuyay-manifest>"}],
            }
        ]
    }
    result = check_inbound_for_injected_tags(body)
    assert result == "Rejected: inbound text contains reserved yuyay tags"


def test_escape_xml_attr_escapes_reserved_chars():
    assert _escape_xml_attr('&"<>') == "&amp;&quot;&lt;&gt;"


def test_eviction_key_returns_path_for_read_entries():
    entry = SimpleNamespace(tool_name="Read", tool_input={"file_path": "src/app.py"})
    assert _eviction_key_for_entry(entry) == "src/app.py"


def test_eviction_key_returns_none_for_non_read_entries():
    entry = SimpleNamespace(tool_name="ToolCall", tool_input={"file_path": "src/app.py"})
    assert _eviction_key_for_entry(entry) is None


def test_inject_system_status_includes_manifest_when_tensors(monkeypatch):
    _patch_pressure(monkeypatch, "low")
    page_store = _make_page_store()
    body = _make_body()
    inject_system_status(body, ts={}, cap=0, request_time=REQUEST_TIME, block_store=None, page_store=page_store)
    content = body["messages"][-1]["content"]
    assert "<yuyay-manifest>" in content


def test_inject_system_status_includes_query_only_high_pressure(monkeypatch):
    _patch_pressure(monkeypatch, "high")
    page_store = _make_page_store()
    block_store = _make_block_store()
    body = _make_body()
    inject_system_status(
        body,
        ts={},
        cap=0,
        request_time=REQUEST_TIME,
        block_store=block_store,
        page_store=page_store,
    )
    content = body["messages"][-1]["content"]
    assert "<yuyay-query>" in content


@pytest.mark.parametrize("pressure", ["low", "moderate"])
def test_inject_system_status_omits_query_when_not_high(monkeypatch, pressure):
    _patch_pressure(monkeypatch, pressure)
    page_store = _make_page_store()
    block_store = _make_block_store()
    body = _make_body()
    inject_system_status(
        body,
        ts={},
        cap=0,
        request_time=REQUEST_TIME,
        block_store=block_store,
        page_store=page_store,
    )
    content = body["messages"][-1]["content"]
    assert "<yuyay-query>" not in content
