"""Tests for pichay.replay — offline replay of context paging on proxy logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pichay.pager import PageEntry, PageStore
from pichay.replay import (
    ReplayTurn,
    SessionReplay,
    _apply_evictions,
    main,
    print_aggregate,
    print_session_report,
    replay_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path: Path, name: str, records: list[dict]) -> Path:
    """Write a list of dicts as JSONL and return the path."""
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return p


def _tool_use(tid: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _tool_result(tid: str, content: str | list, is_error: bool = False) -> dict:
    d: dict = {"type": "tool_result", "tool_use_id": tid, "content": content}
    if is_error:
        d["is_error"] = True
    return d


def _req(messages_full: list[dict], ts: str = "2026-03-01T12:00:00Z") -> dict:
    return {"type": "request", "timestamp": ts, "messages_full": messages_full}


# Content large enough to exceed min_size=500 default.
BIG = "A" * 800

# Content too small to be evicted with default min_size=500.
SMALL = "x" * 50


# ---------------------------------------------------------------------------
# (1) Empty log file
# ---------------------------------------------------------------------------

class TestEmptyLog:
    def test_returns_zero_turns(self, tmp_path):
        p = _write_jsonl(tmp_path, "empty.jsonl", [])
        s = replay_session(p)
        assert s.total_turns == 0
        assert s.total_evictions == 0
        assert s.total_bytes_saved == 0
        assert s.total_bytes_original == 0
        assert s.total_faults == 0
        assert s.turns == []

    def test_properties_safe_on_empty(self, tmp_path):
        p = _write_jsonl(tmp_path, "empty.jsonl", [])
        s = replay_session(p)
        assert s.fault_rate == 0.0
        assert s.reduction_pct == 0.0

    def test_log_path_stored(self, tmp_path):
        p = _write_jsonl(tmp_path, "empty.jsonl", [])
        s = replay_session(p)
        assert s.log_path == str(p)


# ---------------------------------------------------------------------------
# (2) Log with no messages_full field (pre-replay logs)
# ---------------------------------------------------------------------------

class TestNoMessagesFull:
    def test_pre_replay_log_skipped(self, tmp_path):
        """Older proxy logs that lack messages_full produce no turns."""
        records = [
            {
                "type": "request",
                "timestamp": "2026-03-01T12:00:00Z",
                "model": "claude-opus-4-6",
                "messages": {"tool_result_count": 5},
            },
            {
                "type": "response",
                "timestamp": "2026-03-01T12:00:01Z",
                "usage": {"input_tokens": 1000},
            },
        ]
        p = _write_jsonl(tmp_path, "old.jsonl", records)
        s = replay_session(p)
        assert s.total_turns == 0
        assert s.turns == []

    def test_non_request_records_ignored(self, tmp_path):
        """Response, compaction, and other record types are not processed."""
        records = [
            {"type": "response", "timestamp": "T1", "messages_full": []},
            {"type": "compaction", "timestamp": "T2"},
            {"type": "page_faults", "timestamp": "T3"},
        ]
        p = _write_jsonl(tmp_path, "misc.jsonl", records)
        s = replay_session(p)
        assert s.total_turns == 0

    def test_mixed_records_only_eligible_processed(self, tmp_path):
        """Only request records WITH messages_full are counted as turns."""
        records = [
            {"type": "request", "timestamp": "T1"},  # no messages_full
            {"type": "response", "timestamp": "T2"},
            {
                "type": "request",
                "timestamp": "T3",
                "messages_full": [
                    {"role": "user", "content": "hello"},
                ],
            },
        ]
        p = _write_jsonl(tmp_path, "mixed.jsonl", records)
        s = replay_session(p)
        assert s.total_turns == 1
        assert len(s.turns) == 1


# ---------------------------------------------------------------------------
# (3) Single-turn session (too few turns for compaction)
# ---------------------------------------------------------------------------

class TestSingleTurn:
    def test_no_eviction_default_threshold(self, tmp_path):
        """One API call with 2 user turns and default age_threshold=4
        produces no evictions (max turns_from_end = 1 < 4)."""
        messages = [
            {"role": "user", "content": "read /foo.py"},
            {"role": "assistant", "content": [
                _tool_use("tu_1", "Read", {"file_path": "/foo.py"})
            ]},
            {"role": "user", "content": [_tool_result("tu_1", BIG)]},
        ]
        p = _write_jsonl(tmp_path, "single.jsonl", [_req(messages)])
        s = replay_session(p)

        assert s.total_turns == 1
        assert s.total_evictions == 0
        assert s.total_faults == 0
        assert len(s.turns) == 1
        t = s.turns[0]
        assert t.turn == 1
        assert t.evictions == 0
        assert t.faults == 0
        assert t.bytes_saved == 0
        assert t.bytes_original == t.bytes_compacted

    def test_small_results_never_evicted(self, tmp_path):
        """Even if old enough, results below min_size are kept."""
        # 4 user turns with age_threshold=2: turn 1 has tfend=3 (string, skip),
        # turn 2 has tfend=2 (eligible), but content is SMALL → skipped_small.
        messages = [
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": [
                _tool_use("tu_1", "Read", {"file_path": "/tiny.py"})
            ]},
            {"role": "user", "content": [_tool_result("tu_1", SMALL)]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": [{"type": "text", "text": "sure"}]},
            {"role": "user", "content": "done"},
        ]
        p = _write_jsonl(tmp_path, "small.jsonl", [_req(messages)])
        s = replay_session(p, age_threshold=2)
        assert s.total_evictions == 0

    def test_error_results_never_evicted(self, tmp_path):
        """Error tool results are always preserved, regardless of age."""
        messages = [
            {"role": "user", "content": "try"},
            {"role": "assistant", "content": [
                _tool_use("tu_1", "Bash", {"command": "false"})
            ]},
            {"role": "user", "content": [
                _tool_result("tu_1", BIG, is_error=True)
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "failed"}]},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": [{"type": "text", "text": "sure"}]},
            {"role": "user", "content": "done"},
        ]
        p = _write_jsonl(tmp_path, "error.jsonl", [_req(messages)])
        s = replay_session(p, age_threshold=2)
        assert s.total_evictions == 0


# ---------------------------------------------------------------------------
# (4) Multi-turn session where tool results age past threshold
# ---------------------------------------------------------------------------

def _multi_turn_request_1() -> dict:
    """First API call: 4 user turns. With age_threshold=2, the tool_result
    in user turn 2 (tu_1) is old enough to be evicted (turns_from_end=2)."""
    messages = [
        {"role": "user", "content": "read foo"},
        {"role": "assistant", "content": [
            _tool_use("tu_1", "Read", {"file_path": "/foo.py"})
        ]},
        {"role": "user", "content": [_tool_result("tu_1", BIG)]},
        {"role": "assistant", "content": [{"type": "text", "text": "got it"}]},
        {"role": "user", "content": "read bar"},
        {"role": "assistant", "content": [
            _tool_use("tu_2", "Read", {"file_path": "/bar.py"})
        ]},
        {"role": "user", "content": [_tool_result("tu_2", BIG)]},
    ]
    return _req(messages, ts="2026-03-01T12:00:00Z")


def _multi_turn_request_2() -> dict:
    """Second API call: 6 user turns. tu_1 was evicted in request 1.
    Now tu_2 (user turn 4, turns_from_end=2) also gets evicted."""
    messages = [
        {"role": "user", "content": "read foo"},
        {"role": "assistant", "content": [
            _tool_use("tu_1", "Read", {"file_path": "/foo.py"})
        ]},
        {"role": "user", "content": [_tool_result("tu_1", BIG)]},
        {"role": "assistant", "content": [{"type": "text", "text": "got it"}]},
        {"role": "user", "content": "read bar"},
        {"role": "assistant", "content": [
            _tool_use("tu_2", "Read", {"file_path": "/bar.py"})
        ]},
        {"role": "user", "content": [_tool_result("tu_2", BIG)]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": "do stuff"},
        {"role": "assistant", "content": [
            _tool_use("tu_5", "Grep", {"pattern": "TODO", "path": "."})
        ]},
        {"role": "user", "content": [_tool_result("tu_5", BIG)]},
    ]
    return _req(messages, ts="2026-03-01T12:01:00Z")


class TestMultiTurnEviction:
    def test_first_request_evicts_oldest(self, tmp_path):
        """Request 1: 4 user turns, threshold=2, tu_1 evicted."""
        p = _write_jsonl(tmp_path, "multi.jsonl", [_multi_turn_request_1()])
        s = replay_session(p, age_threshold=2)

        assert s.total_turns == 1
        assert s.total_evictions == 1
        t = s.turns[0]
        assert t.evictions == 1
        assert t.bytes_saved > 0

    def test_cumulative_evictions_across_requests(self, tmp_path):
        """Two requests: tu_1 evicted in req 1, tu_2 evicted in req 2."""
        records = [_multi_turn_request_1(), _multi_turn_request_2()]
        p = _write_jsonl(tmp_path, "multi2.jsonl", records)
        s = replay_session(p, age_threshold=2)

        assert s.total_turns == 2
        assert s.total_evictions == 2
        assert s.turns[0].evictions == 1
        assert s.turns[0].cumulative_evictions == 1
        assert s.turns[1].cumulative_evictions == 2

    def test_bytes_saved_increases(self, tmp_path):
        """Total bytes saved grows with each eviction."""
        records = [_multi_turn_request_1(), _multi_turn_request_2()]
        p = _write_jsonl(tmp_path, "multi3.jsonl", records)
        s = replay_session(p, age_threshold=2)

        assert s.total_bytes_saved > 0
        # Each eviction replaces ~800 bytes of content with a short summary
        assert s.turns[0].bytes_saved > 0
        assert s.turns[1].bytes_saved > 0

    def test_reduction_pct(self, tmp_path):
        """Reduction percentage is correctly computed."""
        records = [_multi_turn_request_1(), _multi_turn_request_2()]
        p = _write_jsonl(tmp_path, "multi4.jsonl", records)
        s = replay_session(p, age_threshold=2)

        # Session-level reduction
        assert s.reduction_pct > 0.0
        expected_pct = (s.total_bytes_saved / s.total_bytes_original) * 100
        assert abs(s.reduction_pct - expected_pct) < 0.01

    def test_apply_evictions_replaces_previously_evicted(self, tmp_path):
        """Request 2 should see tu_1's content replaced by _apply_evictions
        before compact_messages runs, since tu_1 was evicted in request 1."""
        records = [_multi_turn_request_1(), _multi_turn_request_2()]
        p = _write_jsonl(tmp_path, "multi5.jsonl", records)
        s = replay_session(p, age_threshold=2)

        # Request 2: bytes_original is from the raw log (includes full tu_1
        # content), but bytes_compacted reflects both _apply_evictions on tu_1
        # AND the new compaction of tu_2.  So bytes_saved on request 2 should
        # be larger than on request 1 (or at least comparable).
        assert s.turns[1].bytes_saved > 0

    def test_custom_thresholds(self, tmp_path):
        """Higher age_threshold means fewer evictions."""
        records = [_multi_turn_request_1()]
        p = _write_jsonl(tmp_path, "threshold.jsonl", records)

        # With age_threshold=2: tu_1 evicted (4 user turns, tfend=2)
        s2 = replay_session(p, age_threshold=2)
        assert s2.total_evictions == 1

        # With age_threshold=4: tu_1 NOT evicted (tfend=2 < 4)
        s4 = replay_session(p, age_threshold=4)
        assert s4.total_evictions == 0

    def test_custom_min_size(self, tmp_path):
        """Higher min_size prevents smaller results from being evicted."""
        records = [_multi_turn_request_1()]
        p = _write_jsonl(tmp_path, "minsize.jsonl", records)

        # BIG is 800 bytes.  min_size=500 → evicted.
        s_lo = replay_session(p, age_threshold=2, min_size=500)
        assert s_lo.total_evictions == 1

        # min_size=1000 → BIG (800) is too small → not evicted.
        s_hi = replay_session(p, age_threshold=2, min_size=1000)
        assert s_hi.total_evictions == 0


# ---------------------------------------------------------------------------
# (5) Page fault detection when model re-requests evicted content
# ---------------------------------------------------------------------------

def _fault_request_2() -> dict:
    """Second request that includes a re-read of /foo.py (tu_3) after
    tu_1 (Read /foo.py) was evicted in request 1. tu_3 is a page fault."""
    messages = [
        {"role": "user", "content": "read foo"},
        {"role": "assistant", "content": [
            _tool_use("tu_1", "Read", {"file_path": "/foo.py"})
        ]},
        {"role": "user", "content": [_tool_result("tu_1", BIG)]},
        {"role": "assistant", "content": [{"type": "text", "text": "got it"}]},
        {"role": "user", "content": "read bar"},
        {"role": "assistant", "content": [
            _tool_use("tu_2", "Read", {"file_path": "/bar.py"})
        ]},
        {"role": "user", "content": [_tool_result("tu_2", BIG)]},
        # The model re-requests /foo.py (page fault)
        {"role": "assistant", "content": [
            _tool_use("tu_3", "Read", {"file_path": "/foo.py"})
        ]},
        {"role": "user", "content": [_tool_result("tu_3", "re-read content")]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": [
            _tool_use("tu_6", "Grep", {"pattern": "TODO", "path": "."})
        ]},
        {"role": "user", "content": [_tool_result("tu_6", BIG)]},
    ]
    return _req(messages, ts="2026-03-01T12:01:00Z")


class TestPageFaultDetection:
    def test_fault_detected(self, tmp_path):
        """A re-read of /foo.py after tu_1 eviction is a page fault."""
        records = [_multi_turn_request_1(), _fault_request_2()]
        p = _write_jsonl(tmp_path, "fault.jsonl", records)
        s = replay_session(p, age_threshold=2)

        # Request 1: tu_1 evicted, no faults.
        assert s.turns[0].faults == 0
        # Request 2: tu_3 re-reads /foo.py → fault detected.
        assert s.turns[1].faults == 1
        assert s.total_faults == 1

    def test_fault_rate(self, tmp_path):
        """fault_rate = total_faults / total_evictions."""
        records = [_multi_turn_request_1(), _fault_request_2()]
        p = _write_jsonl(tmp_path, "faultrate.jsonl", records)
        s = replay_session(p, age_threshold=2)

        assert s.total_evictions > 0
        expected = s.total_faults / s.total_evictions
        assert abs(s.fault_rate - expected) < 1e-9

    def test_cumulative_faults(self, tmp_path):
        """cumulative_faults tracks running total across turns."""
        records = [_multi_turn_request_1(), _fault_request_2()]
        p = _write_jsonl(tmp_path, "cumfault.jsonl", records)
        s = replay_session(p, age_threshold=2)

        assert s.turns[0].cumulative_faults == 0
        assert s.turns[1].cumulative_faults == 1

    def test_original_tool_use_not_counted_as_fault(self, tmp_path):
        """The original tool_use (tu_1) that produced the evicted result
        should NOT be counted as a fault even though it matches."""
        records = [_multi_turn_request_1()]
        p = _write_jsonl(tmp_path, "nofault.jsonl", records)
        s = replay_session(p, age_threshold=2)

        # tu_1 is in the assistant message AND was evicted, but it's the
        # original call — not a re-request. No fault.
        assert s.total_faults == 0

    def test_no_double_counting_across_requests(self, tmp_path):
        """The same fault tool_use_id appearing in multiple request records
        should only be counted once."""
        req1 = _multi_turn_request_1()
        req2 = _fault_request_2()
        # Request 3 is a superset of request 2 (session has grown).
        # tu_3 (the fault) is still in the messages.
        msgs3 = json.loads(json.dumps(req2["messages_full"]))
        msgs3.extend([
            {"role": "assistant", "content": [{"type": "text", "text": "more"}]},
            {"role": "user", "content": "keep going"},
        ])
        req3 = _req(msgs3, ts="2026-03-01T12:02:00Z")

        p = _write_jsonl(tmp_path, "dedup.jsonl", [req1, req2, req3])
        s = replay_session(p, age_threshold=2)

        # tu_3 appears in both req2 and req3, but only counted once.
        assert s.total_faults == 1


# ---------------------------------------------------------------------------
# (6) Byte counting accuracy
# ---------------------------------------------------------------------------

class TestByteCountingAccuracy:
    def test_bytes_original_matches_json_serialization(self, tmp_path):
        """bytes_original should equal len(json.dumps(messages).encode())."""
        messages = [
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": [
                _tool_use("tu_1", "Read", {"file_path": "/foo.py"})
            ]},
            {"role": "user", "content": [_tool_result("tu_1", BIG)]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
            {"role": "user", "content": "done"},
        ]
        expected_bytes = len(json.dumps(messages).encode("utf-8"))
        p = _write_jsonl(tmp_path, "bytes.jsonl", [_req(messages)])
        s = replay_session(p, age_threshold=2)

        assert s.turns[0].bytes_original == expected_bytes

    def test_bytes_saved_equals_diff(self, tmp_path):
        """bytes_saved must equal bytes_original - bytes_compacted."""
        messages = [
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": [
                _tool_use("tu_1", "Read", {"file_path": "/foo.py"})
            ]},
            {"role": "user", "content": [_tool_result("tu_1", BIG)]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
            {"role": "user", "content": "done"},
        ]
        p = _write_jsonl(tmp_path, "bytediff.jsonl", [_req(messages)])
        s = replay_session(p, age_threshold=2)

        t = s.turns[0]
        assert t.bytes_saved == t.bytes_original - t.bytes_compacted

    def test_compacted_smaller_than_original(self, tmp_path):
        """After eviction, compacted bytes should be strictly smaller."""
        messages = [
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": [
                _tool_use("tu_1", "Read", {"file_path": "/foo.py"})
            ]},
            {"role": "user", "content": [_tool_result("tu_1", BIG)]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
            {"role": "user", "content": "done"},
        ]
        p = _write_jsonl(tmp_path, "smaller.jsonl", [_req(messages)])
        s = replay_session(p, age_threshold=2)

        t = s.turns[0]
        assert t.bytes_compacted < t.bytes_original
        assert t.bytes_saved > 0

    def test_no_eviction_bytes_unchanged(self, tmp_path):
        """When nothing is evicted, bytes_original == bytes_compacted."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hey"}]},
            {"role": "user", "content": "bye"},
        ]
        p = _write_jsonl(tmp_path, "nochange.jsonl", [_req(messages)])
        s = replay_session(p)

        t = s.turns[0]
        assert t.bytes_saved == 0
        assert t.bytes_original == t.bytes_compacted

    def test_total_bytes_accumulate(self, tmp_path):
        """Session totals equal the sum of per-turn values."""
        records = [_multi_turn_request_1(), _multi_turn_request_2()]
        p = _write_jsonl(tmp_path, "accum.jsonl", records)
        s = replay_session(p, age_threshold=2)

        assert s.total_bytes_saved == sum(t.bytes_saved for t in s.turns)
        assert s.total_bytes_original == sum(t.bytes_original for t in s.turns)

    def test_reduction_pct_per_turn(self, tmp_path):
        """Per-turn reduction_pct is consistent with byte values."""
        messages = [
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": [
                _tool_use("tu_1", "Read", {"file_path": "/foo.py"})
            ]},
            {"role": "user", "content": [_tool_result("tu_1", BIG)]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
            {"role": "user", "content": "done"},
        ]
        p = _write_jsonl(tmp_path, "pct.jsonl", [_req(messages)])
        s = replay_session(p, age_threshold=2)

        t = s.turns[0]
        expected_pct = t.bytes_saved / t.bytes_original * 100
        assert abs(t.reduction_pct - expected_pct) < 0.01

    def test_message_count_matches(self, tmp_path):
        """ReplayTurn.message_count reflects the original messages_full length."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "bye"},
        ]
        p = _write_jsonl(tmp_path, "count.jsonl", [_req(messages)])
        s = replay_session(p)

        assert s.turns[0].message_count == 3


# ---------------------------------------------------------------------------
# (7) CLI entry point with --json flag
# ---------------------------------------------------------------------------

class TestCLIEntryPoint:
    def test_json_single_session(self, tmp_path, capsys, monkeypatch):
        """--json outputs one JSON line per session."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "bye"},
        ]
        p = _write_jsonl(tmp_path, "cli.jsonl", [_req(messages)])

        monkeypatch.setattr(
            sys, "argv",
            ["replay", "--json", str(p)],
        )
        main()

        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert data["log_path"] == str(p)
        assert data["total_turns"] == 1
        assert "fault_rate" in data
        assert "reduction_pct" in data
        assert "turn_count" in data
        # turns list should be replaced by turn_count
        assert "turns" not in data

    def test_json_multiple_sessions_includes_aggregate(self, tmp_path, capsys, monkeypatch):
        """With multiple logs and --json, the last line is an aggregate."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "bye"},
        ]
        p1 = _write_jsonl(tmp_path, "s1.jsonl", [_req(msgs)])
        p2 = _write_jsonl(tmp_path, "s2.jsonl", [_req(msgs)])

        monkeypatch.setattr(
            sys, "argv",
            ["replay", "--json", str(p1), str(p2)],
        )
        main()

        lines = capsys.readouterr().out.strip().split("\n")
        assert len(lines) == 3  # 2 sessions + 1 aggregate

        s1 = json.loads(lines[0])
        s2 = json.loads(lines[1])
        agg = json.loads(lines[2])

        assert agg["type"] == "aggregate"
        assert agg["sessions"] == 2
        assert agg["total_turns"] == s1["total_turns"] + s2["total_turns"]

    def test_json_custom_thresholds(self, tmp_path, capsys, monkeypatch):
        """--age-threshold and --min-size are forwarded to replay_session."""
        records = [_multi_turn_request_1()]
        p = _write_jsonl(tmp_path, "custom.jsonl", records)

        monkeypatch.setattr(
            sys, "argv",
            ["replay", "--json", "--age-threshold", "2", "--min-size", "100", str(p)],
        )
        main()

        data = json.loads(capsys.readouterr().out.strip())
        assert data["total_evictions"] == 1

    def test_missing_file_skipped(self, tmp_path, capsys, monkeypatch):
        """Non-existent log files emit a warning and are skipped."""
        fake = tmp_path / "does_not_exist.jsonl"

        monkeypatch.setattr(
            sys, "argv",
            ["replay", "--json", str(fake)],
        )
        main()

        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "warning" in captured.err.lower()
        # No JSON output for a missing file
        assert captured.out.strip() == ""

    def test_human_readable_output(self, tmp_path, capsys, monkeypatch):
        """Without --json, output is human-readable (contains '=' banner)."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hey"}]},
            {"role": "user", "content": "bye"},
        ]
        p = _write_jsonl(tmp_path, "human.jsonl", [_req(msgs)])

        monkeypatch.setattr(
            sys, "argv",
            ["replay", str(p)],
        )
        main()

        out = capsys.readouterr().out
        assert "Replay:" in out
        assert "Turns:" in out


# ---------------------------------------------------------------------------
# (8) Aggregate stats across multiple sessions
# ---------------------------------------------------------------------------

class TestAggregateStats:
    def _make_two_sessions(self, tmp_path):
        """Create two log files with known eviction/fault patterns."""
        # Session A: 1 eviction, 0 faults
        p_a = _write_jsonl(
            tmp_path, "a.jsonl", [_multi_turn_request_1()],
        )
        # Session B: 2 evictions, 1 fault (re-read of /foo.py)
        p_b = _write_jsonl(
            tmp_path, "b.jsonl",
            [_multi_turn_request_1(), _fault_request_2()],
        )
        return p_a, p_b

    def test_aggregate_totals(self, tmp_path):
        """Aggregate totals equal the sum of per-session values."""
        p_a, p_b = self._make_two_sessions(tmp_path)
        s_a = replay_session(p_a, age_threshold=2)
        s_b = replay_session(p_b, age_threshold=2)
        sessions = [s_a, s_b]

        total_turns = sum(s.total_turns for s in sessions)
        total_evictions = sum(s.total_evictions for s in sessions)
        total_bytes_saved = sum(s.total_bytes_saved for s in sessions)
        total_bytes_original = sum(s.total_bytes_original for s in sessions)
        total_faults = sum(s.total_faults for s in sessions)

        assert total_turns == s_a.total_turns + s_b.total_turns
        assert total_evictions == s_a.total_evictions + s_b.total_evictions
        assert total_bytes_saved == s_a.total_bytes_saved + s_b.total_bytes_saved
        assert total_bytes_original == (
            s_a.total_bytes_original + s_b.total_bytes_original
        )
        assert total_faults == s_a.total_faults + s_b.total_faults

    def test_aggregate_fault_rate(self, tmp_path):
        """Aggregate fault_rate = total_faults / total_evictions."""
        p_a, p_b = self._make_two_sessions(tmp_path)
        s_a = replay_session(p_a, age_threshold=2)
        s_b = replay_session(p_b, age_threshold=2)

        total_e = s_a.total_evictions + s_b.total_evictions
        total_f = s_a.total_faults + s_b.total_faults
        expected_rate = total_f / total_e if total_e > 0 else 0.0

        assert total_e > 0
        assert abs(expected_rate - total_f / total_e) < 1e-9

    def test_aggregate_json_output(self, tmp_path, capsys, monkeypatch):
        """--json with multiple logs produces per-session + aggregate lines."""
        p_a, p_b = self._make_two_sessions(tmp_path)

        monkeypatch.setattr(
            sys, "argv",
            ["replay", "--json", "--age-threshold", "2", str(p_a), str(p_b)],
        )
        main()

        lines = capsys.readouterr().out.strip().split("\n")
        assert len(lines) == 3

        sa = json.loads(lines[0])
        sb = json.loads(lines[1])
        agg = json.loads(lines[2])

        assert agg["type"] == "aggregate"
        assert agg["sessions"] == 2
        assert agg["total_evictions"] == sa["total_evictions"] + sb["total_evictions"]
        assert agg["total_bytes_saved"] == (
            sa["total_bytes_saved"] + sb["total_bytes_saved"]
        )
        assert agg["total_faults"] == sa["total_faults"] + sb["total_faults"]
        assert agg["total_bytes_original"] == (
            sa["total_bytes_original"] + sb["total_bytes_original"]
        )
        # Derived fields
        assert "fault_rate" in agg
        assert "reduction_pct" in agg

    def test_print_aggregate_no_crash(self, tmp_path, capsys):
        """print_aggregate runs without error on valid sessions."""
        p_a, p_b = self._make_two_sessions(tmp_path)
        sessions = [
            replay_session(p_a, age_threshold=2),
            replay_session(p_b, age_threshold=2),
        ]
        # Should not raise
        print_aggregate(sessions)
        out = capsys.readouterr().out
        assert "AGGREGATE" in out

    def test_print_aggregate_empty(self, capsys):
        """print_aggregate with no sessions produces no output."""
        print_aggregate([])
        assert capsys.readouterr().out == ""

    def test_print_session_report_no_crash(self, tmp_path, capsys):
        """print_session_report runs cleanly on a session with turns."""
        records = [_multi_turn_request_1(), _multi_turn_request_2()]
        p = _write_jsonl(tmp_path, "report.jsonl", records)
        s = replay_session(p, age_threshold=2)
        print_session_report(s)
        out = capsys.readouterr().out
        assert "Replay:" in out
        assert "Per-turn:" in out

    def test_print_session_report_empty_session(self, tmp_path, capsys):
        """print_session_report on a session with no turns is safe."""
        p = _write_jsonl(tmp_path, "empty.jsonl", [])
        s = replay_session(p)
        print_session_report(s)
        out = capsys.readouterr().out
        assert "Replay:" in out
        # Should NOT have per-turn output
        assert "Per-turn:" not in out


# ---------------------------------------------------------------------------
# Edge cases and _apply_evictions unit test
# ---------------------------------------------------------------------------

class TestApplyEvictions:
    def test_replaces_evicted_tool_result(self):
        """Directly test _apply_evictions replaces known evicted content."""
        store = PageStore()
        entry = PageEntry(
            tool_use_id="tu_99",
            tool_name="Read",
            tool_input={"file_path": "/evicted.py"},
            original_content="original data here",
            original_size=100,
            summary="[Paged out]",
            evicted_at=0.0,
            turn_index=1,
            turns_from_end=5,
        )
        store.pages["tu_99"] = entry

        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_99",
                 "content": "original data here"},
            ]},
        ]
        _apply_evictions(messages, store)

        block = messages[0]["content"][0]
        assert block["content"] == "[Paged out]"
        assert block["tool_use_id"] == "tu_99"

    def test_skips_non_user_messages(self):
        """_apply_evictions only touches user messages."""
        store = PageStore()
        store.pages["tu_1"] = PageEntry(
            tool_use_id="tu_1", tool_name="Read",
            tool_input={}, original_content="x",
            original_size=1, summary="[paged]",
            evicted_at=0.0, turn_index=1, turns_from_end=5,
        )

        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Read",
                 "input": {"file_path": "/f.py"}},
            ]},
        ]
        _apply_evictions(messages, store)
        # Assistant message should be unchanged
        assert messages[0]["content"][0]["type"] == "tool_use"

    def test_skips_string_content(self):
        """User messages with string content (not list) are skipped."""
        store = PageStore()
        messages = [
            {"role": "user", "content": "just a string"},
        ]
        _apply_evictions(messages, store)
        assert messages[0]["content"] == "just a string"

    def test_leaves_non_evicted_alone(self):
        """Tool results not in the page store are left unchanged."""
        store = PageStore()
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_unknown",
                 "content": "keep me"},
            ]},
        ]
        _apply_evictions(messages, store)
        assert messages[0]["content"][0]["content"] == "keep me"


# ---------------------------------------------------------------------------
# SessionReplay dataclass properties
# ---------------------------------------------------------------------------

class TestSessionReplayProperties:
    def test_fault_rate_zero_evictions(self):
        s = SessionReplay(log_path="x")
        assert s.fault_rate == 0.0

    def test_fault_rate_with_evictions(self):
        s = SessionReplay(log_path="x", total_evictions=10, total_faults=3)
        assert s.fault_rate == pytest.approx(0.3)

    def test_reduction_pct_zero_bytes(self):
        s = SessionReplay(log_path="x")
        assert s.reduction_pct == 0.0

    def test_reduction_pct_with_bytes(self):
        s = SessionReplay(
            log_path="x",
            total_bytes_original=1000,
            total_bytes_saved=250,
        )
        assert s.reduction_pct == pytest.approx(25.0)
