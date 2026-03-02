"""Tests for pichay.trimmer — system prompt trimming for the proxy."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from pichay.trimmer import (
    SessionTrimReport,
    SkillDedupeStats,
    StaticCacheStats,
    SystemPromptTrimmer,
    ToolStubStats,
    TrimResult,
    TrimTurn,
    _STUB_SCHEMA,
    _content_hash,
    _dedupe_skills_text,
    _make_tool_stub,
    analyze_trimming,
    main,
    print_trim_report,
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


def _tool_def(name: str, desc: str = "A tool.", schema: dict | None = None) -> dict:
    """Create a tool definition dict."""
    return {
        "name": name,
        "description": desc,
        "input_schema": schema or {
            "type": "object",
            "properties": {"arg": {"type": "string"}},
            "required": ["arg"],
        },
    }


def _tool_use(tid: str, name: str, inp: dict | None = None) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp or {}}


def _tool_result(tid: str, content: str = "ok") -> dict:
    return {"type": "tool_result", "tool_use_id": tid, "content": content}


def _skill_reminder(entries: list[str]) -> str:
    """Build a <system-reminder> block with a skill list."""
    lines = "\n".join(f"- {e}" for e in entries)
    return (
        "<system-reminder>\n"
        "The following skills are available for use with the Skill tool:\n\n"
        f"{lines}\n"
        "</system-reminder>"
    )


def _body(
    tools: list[dict] | None = None,
    messages: list[dict] | None = None,
    system: list[dict] | str | None = None,
) -> dict:
    """Build a minimal request body."""
    b: dict = {}
    if tools is not None:
        b["tools"] = tools
    if messages is not None:
        b["messages"] = messages
    else:
        b["messages"] = []
    if system is not None:
        b["system"] = system
    return b


# ---------------------------------------------------------------------------
# (1) Tool definition trimming
# ---------------------------------------------------------------------------

class TestToolTrimFirstRequest:
    """First request with no prior tool usage — all tools should be stubbed."""

    def test_all_tools_stubbed(self):
        trimmer = SystemPromptTrimmer()
        tools = [_tool_def("Read"), _tool_def("Write"), _tool_def("Bash")]
        body = _body(tools=tools)
        result = trimmer.trim(body)

        assert result.tools.total_tools == 3
        assert result.tools.stubbed_tools == 3
        assert result.tools.restored_tools == 0

    def test_stub_format(self):
        trimmer = SystemPromptTrimmer()
        tools = [_tool_def("Read", desc="Reads a file.\nMore details here.")]
        body = _body(tools=tools)
        trimmer.trim(body)

        stub = body["tools"][0]
        assert stub["name"] == "Read"
        assert stub["input_schema"] == _STUB_SCHEMA
        # First line only, no newline content
        assert "\n" not in stub["description"]

    def test_bytes_before_gt_bytes_after(self):
        trimmer = SystemPromptTrimmer()
        tools = [_tool_def("Read", desc="A" * 200, schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The file"},
                "offset": {"type": "number"},
                "limit": {"type": "number"},
            },
            "required": ["file_path"],
        })]
        body = _body(tools=tools)
        result = trimmer.trim(body)

        assert result.tools.bytes_before > result.tools.bytes_after
        assert result.tools.bytes_saved > 0


class TestToolTrimAfterUsage:
    """After a tool is used, its full definition should be restored."""

    def test_used_tool_restored_others_stubbed(self):
        trimmer = SystemPromptTrimmer()
        tools = [_tool_def("Read"), _tool_def("Write"), _tool_def("Bash")]

        # First request: no usage yet, all stubbed
        body1 = _body(tools=tools[:])
        trimmer.trim(body1)

        # Second request: assistant used "Read" in messages
        messages = [
            {"role": "user", "content": "read foo"},
            {"role": "assistant", "content": [_tool_use("tu_1", "Read")]},
            {"role": "user", "content": [_tool_result("tu_1")]},
        ]
        body2 = _body(tools=tools[:], messages=messages)
        result2 = trimmer.trim(body2)

        # Read should be full, Write and Bash should be stubbed
        assert result2.tools.stubbed_tools == 2
        tool_map = {t["name"]: t for t in body2["tools"]}
        assert tool_map["Read"]["input_schema"] != _STUB_SCHEMA
        assert tool_map["Write"]["input_schema"] == _STUB_SCHEMA
        assert tool_map["Bash"]["input_schema"] == _STUB_SCHEMA

    def test_restored_tools_count(self):
        """restored_tools counts tools that were stubs and got re-injected."""
        trimmer = SystemPromptTrimmer()
        full_schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        tools = [_tool_def("Read", schema=full_schema)]

        # First request: store the full definition, stub it
        body1 = _body(tools=[t.copy() for t in tools])
        trimmer.trim(body1)
        # The tool is now a stub in body1
        assert body1["tools"][0]["input_schema"] == _STUB_SCHEMA

        # Second request: tool was used, and the incoming body has stubs
        messages = [
            {"role": "assistant", "content": [_tool_use("tu_1", "Read")]},
        ]
        # Send the stub version (as the API would after our previous trimming)
        body2 = _body(tools=[body1["tools"][0].copy()], messages=messages)
        result2 = trimmer.trim(body2)

        assert result2.tools.restored_tools == 1
        # Full schema should be restored
        assert body2["tools"][0]["input_schema"] == full_schema

    def test_full_defs_stored_on_first_encounter(self):
        trimmer = SystemPromptTrimmer()
        schema = {"type": "object", "properties": {"p": {"type": "integer"}}}
        tools = [_tool_def("Glob", schema=schema)]
        body = _body(tools=tools[:])
        trimmer.trim(body)

        assert "Glob" in trimmer._full_tool_defs
        assert trimmer._full_tool_defs["Glob"]["input_schema"] == schema


class TestToolStubFormat:
    """_make_tool_stub produces the correct stub format."""

    def test_first_line_only(self):
        tool = _tool_def("X", desc="Line one.\nLine two.\nLine three.")
        stub = _make_tool_stub(tool)
        assert stub["description"] == "Line one."

    def test_truncation_at_120(self):
        long_desc = "A" * 200
        tool = _tool_def("X", desc=long_desc)
        stub = _make_tool_stub(tool)
        assert len(stub["description"]) == 120
        assert stub["description"].endswith("...")

    def test_dash_prefix_stripped(self):
        tool = _tool_def("X", desc="- Some description starting with dash")
        stub = _make_tool_stub(tool)
        assert stub["description"] == "Some description starting with dash"

    def test_empty_description(self):
        tool = {"name": "X", "description": "", "input_schema": {}}
        stub = _make_tool_stub(tool)
        assert stub["description"] == ""
        assert stub["name"] == "X"
        assert stub["input_schema"] == _STUB_SCHEMA

    def test_schema_is_stub_schema(self):
        tool = _tool_def("X", schema={"type": "object", "properties": {"a": {}}})
        stub = _make_tool_stub(tool)
        assert stub["input_schema"] == _STUB_SCHEMA


# ---------------------------------------------------------------------------
# (2) Skill deduplication
# ---------------------------------------------------------------------------

class TestSkillDeduplication:
    """Skills with multiple prefixes are deduplicated."""

    def test_tripled_skills_deduped(self):
        trimmer = SystemPromptTrimmer()
        reminder = _skill_reminder([
            "pptx: Presentation creation",
            "example-skills:pptx: Presentation creation",
            "document-skills:pptx: Presentation creation",
        ])
        body = _body(messages=[{"role": "user", "content": reminder}])
        result = trimmer.trim(body)

        assert result.skills.total_entries == 3
        assert result.skills.duplicates_removed == 2
        assert result.skills.unique_skills == 1

    def test_no_duplicates_untouched(self):
        trimmer = SystemPromptTrimmer()
        reminder = _skill_reminder([
            "pptx: Presentations",
            "docx: Documents",
            "xlsx: Spreadsheets",
        ])
        body = _body(messages=[{"role": "user", "content": reminder}])
        result = trimmer.trim(body)

        assert result.skills.duplicates_removed == 0
        # When no dupes, bytes_before/after are 0 (no mutation happened)
        assert result.skills.bytes_saved == 0

    def test_string_content(self):
        """Works when message content is a plain string."""
        trimmer = SystemPromptTrimmer()
        reminder = _skill_reminder([
            "pdf: PDF tools",
            "example-skills:pdf: PDF tools",
        ])
        body = _body(messages=[{"role": "user", "content": reminder}])
        result = trimmer.trim(body)

        assert result.skills.duplicates_removed == 1
        assert result.skills.bytes_saved > 0

    def test_list_content_blocks(self):
        """Works when message content is a list of text blocks."""
        trimmer = SystemPromptTrimmer()
        reminder = _skill_reminder([
            "xlsx: Spreadsheets",
            "document-skills:xlsx: Spreadsheets",
        ])
        block = {"type": "text", "text": reminder}
        body = _body(messages=[{"role": "user", "content": [block]}])
        result = trimmer.trim(body)

        assert result.skills.duplicates_removed == 1

    def test_bytes_saved_accurate(self):
        trimmer = SystemPromptTrimmer()
        entries = [
            "pptx: Presentation creation and editing",
            "example-skills:pptx: Presentation creation and editing",
            "document-skills:pptx: Presentation creation and editing",
        ]
        reminder = _skill_reminder(entries)
        body = _body(messages=[{"role": "user", "content": reminder}])

        before_bytes = len(reminder.encode("utf-8"))
        result = trimmer.trim(body)
        after_bytes = len(body["messages"][0]["content"].encode("utf-8"))

        assert result.skills.bytes_saved == before_bytes - after_bytes

    def test_non_skill_reminder_unchanged(self):
        """System reminders without skills are not modified."""
        trimmer = SystemPromptTrimmer()
        content = "<system-reminder>\nUSD budget: $5/$15\n</system-reminder>"
        body = _body(messages=[{"role": "user", "content": content}])
        trimmer.trim(body)
        assert body["messages"][0]["content"] == content


class TestDedupeSkillsText:
    """Unit tests for the _dedupe_skills_text helper."""

    def test_returns_tuple(self):
        text = _skill_reminder(["a: desc", "example-skills:a: desc"])
        new_text, entries, dupes = _dedupe_skills_text(text)
        assert isinstance(new_text, str)
        assert entries == 2
        assert dupes == 1

    def test_no_skills_block(self):
        text = "<system-reminder>\nSome other content\n</system-reminder>"
        new_text, entries, dupes = _dedupe_skills_text(text)
        assert new_text == text
        assert entries == 0
        assert dupes == 0

    def test_mixed_unique_and_duplicate(self):
        text = _skill_reminder([
            "pdf: PDF tools",
            "docx: Document tools",
            "example-skills:pdf: PDF tools",
            "xlsx: Spreadsheet tools",
            "document-skills:docx: Document tools",
        ])
        _, entries, dupes = _dedupe_skills_text(text)
        assert entries == 5
        assert dupes == 2


# ---------------------------------------------------------------------------
# (3) Static component tracking
# ---------------------------------------------------------------------------

class TestStaticComponentTracking:
    """_track_static detects unchanged components across turns."""

    def test_first_turn_nothing_static(self):
        trimmer = SystemPromptTrimmer()
        body = _body(system=[{"type": "text", "text": "You are helpful."}])
        result = trimmer.trim(body)

        assert result.static.total_components == 1
        assert result.static.static_components == 0
        assert result.static.static_bytes_skippable == 0

    def test_second_turn_same_content_all_static(self):
        trimmer = SystemPromptTrimmer()
        sys_blocks = [{"type": "text", "text": "You are helpful."}]

        body1 = _body(system=sys_blocks[:])
        trimmer.trim(body1)

        body2 = _body(system=sys_blocks[:])
        result2 = trimmer.trim(body2)

        assert result2.static.total_components == 1
        assert result2.static.static_components == 1
        assert result2.static.static_bytes_skippable > 0

    def test_changed_component_not_static(self):
        trimmer = SystemPromptTrimmer()

        body1 = _body(system=[{"type": "text", "text": "Version 1"}])
        trimmer.trim(body1)

        body2 = _body(system=[{"type": "text", "text": "Version 2"}])
        result2 = trimmer.trim(body2)

        assert result2.static.static_components == 0

    def test_static_bytes_correct(self):
        trimmer = SystemPromptTrimmer()
        text = "This is a system prompt block with some content."
        sys_blocks = [{"type": "text", "text": text}]

        trimmer.trim(_body(system=sys_blocks[:]))
        result = trimmer.trim(_body(system=sys_blocks[:]))

        expected_bytes = len(text.encode("utf-8"))
        assert result.static.static_bytes_skippable == expected_bytes

    def test_system_reminders_tracked(self):
        trimmer = SystemPromptTrimmer()
        reminder = "<system-reminder>Budget: $5</system-reminder>"
        msgs = [{"role": "user", "content": reminder}]

        trimmer.trim(_body(messages=msgs[:]))
        result = trimmer.trim(_body(messages=msgs[:]))

        assert result.static.static_components == 1

    def test_system_as_string(self):
        trimmer = SystemPromptTrimmer()

        body1 = _body(system="You are a helpful assistant.")
        trimmer.trim(body1)

        body2 = _body(system="You are a helpful assistant.")
        result = trimmer.trim(body2)

        assert result.static.static_components == 1
        assert result.static.static_bytes_skippable == len(
            "You are a helpful assistant.".encode("utf-8")
        )

    def test_multiple_components_mixed(self):
        """Some components change, others stay static."""
        trimmer = SystemPromptTrimmer()
        sys1 = [
            {"type": "text", "text": "Static block"},
            {"type": "text", "text": "Dynamic block v1"},
        ]
        trimmer.trim(_body(system=sys1))

        sys2 = [
            {"type": "text", "text": "Static block"},
            {"type": "text", "text": "Dynamic block v2"},
        ]
        result = trimmer.trim(_body(system=sys2))

        assert result.static.total_components == 2
        assert result.static.static_components == 1


# ---------------------------------------------------------------------------
# (4) Integration — TrimResult, cumulative stats, log_fn
# ---------------------------------------------------------------------------

class TestTrimResult:
    """TrimResult combines all three interventions."""

    def test_total_bytes_saved(self):
        r = TrimResult()
        r.tools = ToolStubStats(bytes_before=1000, bytes_after=200)
        r.skills = SkillDedupeStats(bytes_before=500, bytes_after=300)
        assert r.total_bytes_saved == (800 + 200)

    def test_total_bytes_skippable(self):
        r = TrimResult()
        r.tools = ToolStubStats(bytes_before=1000, bytes_after=200)
        r.skills = SkillDedupeStats(bytes_before=500, bytes_after=300)
        r.static = StaticCacheStats(static_bytes_skippable=150)
        assert r.total_bytes_skippable == (800 + 200 + 150)

    def test_default_result_zero(self):
        r = TrimResult()
        assert r.total_bytes_saved == 0
        assert r.total_bytes_skippable == 0


class TestCumulativeStats:
    """Cumulative stats accumulate across trim() calls."""

    def test_requests_counted(self):
        trimmer = SystemPromptTrimmer()
        for _ in range(3):
            trimmer.trim(_body())
        assert trimmer.cumulative_requests == 3

    def test_tool_bytes_accumulate(self):
        trimmer = SystemPromptTrimmer()
        tools = [_tool_def("Read", desc="A" * 200, schema={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        })]
        trimmer.trim(_body(tools=tools[:]))
        saved_1 = trimmer.cumulative_tools_bytes_saved

        trimmer.trim(_body(tools=tools[:]))
        saved_2 = trimmer.cumulative_tools_bytes_saved

        assert saved_1 > 0
        assert saved_2 == saved_1 * 2

    def test_skills_bytes_accumulate(self):
        trimmer = SystemPromptTrimmer()
        reminder = _skill_reminder([
            "pdf: PDF tools",
            "example-skills:pdf: PDF tools",
        ])

        # Fresh message dicts each call — _dedupe_skills mutates in place
        trimmer.trim(_body(messages=[{"role": "user", "content": reminder}]))
        saved_1 = trimmer.cumulative_skills_bytes_saved

        trimmer.trim(_body(messages=[{"role": "user", "content": reminder}]))
        saved_2 = trimmer.cumulative_skills_bytes_saved

        assert saved_1 > 0
        assert saved_2 == saved_1 * 2


class TestSummary:
    """summary() returns correct state."""

    def test_summary_initial(self):
        trimmer = SystemPromptTrimmer()
        s = trimmer.summary()
        assert s["requests_trimmed"] == 0
        assert s["tools_used"] == []
        assert s["tools_known"] == 0
        assert s["cumulative_tools_bytes_saved"] == 0
        assert s["cumulative_skills_bytes_saved"] == 0

    def test_summary_after_trim(self):
        trimmer = SystemPromptTrimmer()
        tools = [_tool_def("Read"), _tool_def("Write")]
        messages = [
            {"role": "assistant", "content": [_tool_use("t1", "Read")]},
        ]
        trimmer.trim(_body(tools=tools[:], messages=messages))

        s = trimmer.summary()
        assert s["requests_trimmed"] == 1
        assert s["tools_used"] == ["Read"]
        assert s["tools_known"] == 2
        assert s["cumulative_tools_bytes_saved"] > 0


class TestLogFn:
    """log_fn is called with correct record format."""

    def test_log_fn_called(self):
        records: list[dict] = []
        trimmer = SystemPromptTrimmer(log_fn=records.append)
        tools = [_tool_def("Read")]
        trimmer.trim(_body(tools=tools[:]))

        assert len(records) == 1
        rec = records[0]
        assert rec["type"] == "trimming"
        assert rec["request_num"] == 1
        assert "timestamp" in rec
        assert "tools" in rec
        assert "skills" in rec
        assert "static" in rec
        assert "total_bytes_saved" in rec
        assert "cumulative_tools_saved" in rec
        assert "cumulative_skills_saved" in rec

    def test_log_fn_not_called_when_none(self):
        trimmer = SystemPromptTrimmer(log_fn=None)
        # Should not raise
        trimmer.trim(_body(tools=[_tool_def("X")]))

    def test_log_fn_accumulates(self):
        records: list[dict] = []
        trimmer = SystemPromptTrimmer(log_fn=records.append)
        trimmer.trim(_body(tools=[_tool_def("X")]))
        trimmer.trim(_body(tools=[_tool_def("X")]))

        assert len(records) == 2
        assert records[0]["request_num"] == 1
        assert records[1]["request_num"] == 2
        assert records[1]["cumulative_tools_saved"] >= records[0]["cumulative_tools_saved"]


# ---------------------------------------------------------------------------
# (5) Offline analysis
# ---------------------------------------------------------------------------

class TestAnalyzeTrimming:
    """analyze_trimming on proxy logs."""

    def test_empty_log(self, tmp_path):
        p = _write_jsonl(tmp_path, "empty.jsonl", [])
        report = analyze_trimming(p)
        assert report.total_turns == 0
        assert report.turns == []
        assert report.log_path == str(p)

    def test_no_messages_full_graceful(self, tmp_path):
        """Requests without messages_full should be handled gracefully."""
        records = [
            {"type": "request", "timestamp": "2026-03-01T12:00:00Z"},
            {"type": "response", "timestamp": "2026-03-01T12:00:01Z",
             "status_code": 200, "usage": {
                 "input_tokens": 100, "output_tokens": 50}},
        ]
        p = _write_jsonl(tmp_path, "no_msgs.jsonl", records)
        report = analyze_trimming(p)
        # Should not crash; processes the request (messages_full defaults to [])
        assert report.total_turns == 1

    def test_response_only_no_crash(self, tmp_path):
        """Log with only responses (no requests) produces empty report."""
        records = [
            {"type": "response", "timestamp": "2026-03-01T12:00:01Z",
             "status_code": 200, "usage": {
                 "input_tokens": 100, "output_tokens": 50}},
        ]
        p = _write_jsonl(tmp_path, "resp_only.jsonl", records)
        report = analyze_trimming(p)
        assert report.total_turns == 0


class TestSessionTrimReportProperties:
    """SessionTrimReport computed properties."""

    def test_avg_skill_bytes_saved_zero_turns(self):
        r = SessionTrimReport(log_path="test.jsonl")
        assert r.avg_skill_bytes_saved == 0.0

    def test_avg_tool_bytes_saveable_zero_turns(self):
        r = SessionTrimReport(log_path="test.jsonl")
        assert r.avg_tool_bytes_saveable == 0.0

    def test_avg_skill_bytes_saved(self):
        r = SessionTrimReport(log_path="test.jsonl", total_turns=4,
                              total_skill_bytes_saved=1000)
        assert r.avg_skill_bytes_saved == 250.0

    def test_avg_tool_bytes_saveable(self):
        r = SessionTrimReport(log_path="test.jsonl", total_turns=5,
                              total_tool_bytes_saveable=500)
        assert r.avg_tool_bytes_saveable == 100.0


class TestCLIEntryPoint:
    """CLI for print_trim_report and JSON output."""

    def test_print_trim_report(self, capsys):
        r = SessionTrimReport(
            log_path="test.jsonl",
            total_turns=2,
            total_tools_defined=10,
            total_tools_used=3,
            per_request_tool_bytes=5000,
            total_skill_duplicates=4,
            total_skill_bytes_saved=800,
            total_static_bytes_skippable=1200,
            total_tool_bytes_saveable=3000,
        )
        print_trim_report(r)
        out = capsys.readouterr().out

        assert "SYSTEM PROMPT TRIMMING ANALYSIS" in out
        assert "test.jsonl" in out
        assert "Tool Definition Trimming" in out
        assert "Skill Deduplication" in out
        assert "Static Component Caching" in out
        assert "Summary" in out

    def test_print_trim_report_with_turns(self, capsys):
        r = SessionTrimReport(
            log_path="test.jsonl",
            total_turns=1,
            turns=[TrimTurn(turn=1, timestamp="2026-03-01T12:00:00Z",
                            tools_stubbable=5, tool_bytes_saveable=2000,
                            skill_duplicates=2, skill_bytes_saved=400,
                            static_components=3, static_bytes_skippable=600)],
        )
        print_trim_report(r)
        out = capsys.readouterr().out
        assert "Per-turn:" in out
        assert "T  1" in out

    def test_main_json_output(self, tmp_path, monkeypatch, capsys):
        """main() with --json flag produces JSON output."""
        # Create a minimal valid log
        records = [
            {"type": "request", "timestamp": "2026-03-01T12:00:00Z",
             "messages_full": [], "system_prompt_full": []},
            {"type": "response", "timestamp": "2026-03-01T12:00:01Z",
             "status_code": 200, "usage": {
                 "input_tokens": 100, "output_tokens": 50}},
        ]
        p = _write_jsonl(tmp_path, "log.jsonl", records)
        monkeypatch.setattr(sys, "argv", ["trimmer", "--json", str(p)])
        main()
        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert "log_path" in data
        assert "avg_skill_bytes_saved" in data
        assert "avg_tool_bytes_saveable" in data
        assert "turn_count" in data
        assert "turns" not in data  # turns replaced by turn_count

    def test_main_missing_file(self, tmp_path, monkeypatch, capsys):
        """main() warns and skips missing files."""
        monkeypatch.setattr(
            sys, "argv", ["trimmer", str(tmp_path / "missing.jsonl")]
        )
        main()
        err = capsys.readouterr().err
        assert "not found" in err


# ---------------------------------------------------------------------------
# (6) Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for the trimmer."""

    def test_empty_tools_list(self):
        trimmer = SystemPromptTrimmer()
        body = _body(tools=[])
        result = trimmer.trim(body)
        assert result.tools.total_tools == 0
        assert result.tools.bytes_saved == 0

    def test_no_system_reminder_blocks(self):
        trimmer = SystemPromptTrimmer()
        body = _body(messages=[{"role": "user", "content": "Hello"}])
        result = trimmer.trim(body)
        assert result.skills.duplicates_removed == 0
        assert result.static.total_components == 0

    def test_no_system_prompt_in_body(self):
        trimmer = SystemPromptTrimmer()
        body = _body()
        result = trimmer.trim(body)
        assert result.static.total_components == 0

    def test_tool_very_long_description(self):
        long_desc = "X" * 300 + "\nSecond line"
        stub = _make_tool_stub(_tool_def("LongTool", desc=long_desc))
        assert len(stub["description"]) == 120
        assert stub["description"].endswith("...")

    def test_tool_no_description(self):
        tool = {"name": "NoDesc", "input_schema": {"type": "object"}}
        stub = _make_tool_stub(tool)
        assert stub["description"] == ""
        assert stub["name"] == "NoDesc"

    def test_no_tools_key_in_body(self):
        trimmer = SystemPromptTrimmer()
        body = {"messages": []}
        result = trimmer.trim(body)
        assert result.tools.total_tools == 0

    def test_tools_not_a_list(self):
        trimmer = SystemPromptTrimmer()
        body = {"messages": [], "tools": "invalid"}
        result = trimmer.trim(body)
        # tools block guard: isinstance check fails, no tool trimming
        assert result.tools.total_tools == 0

    def test_message_content_not_list_or_str(self):
        """Non-string, non-list content should not crash skill dedup."""
        trimmer = SystemPromptTrimmer()
        body = _body(messages=[{"role": "user", "content": 42}])
        result = trimmer.trim(body)
        assert result.skills.duplicates_removed == 0

    def test_scan_tool_usage_skips_user_messages(self):
        """Only assistant messages with tool_use blocks count as usage."""
        trimmer = SystemPromptTrimmer()
        tools = [_tool_def("Read")]
        messages = [
            {"role": "user", "content": [_tool_use("t1", "Read")]},
        ]
        body = _body(tools=tools[:], messages=messages)
        trimmer.trim(body)

        # "Read" should NOT be in used_tools (it was in a user message)
        assert "Read" not in trimmer._used_tools
        assert body["tools"][0]["input_schema"] == _STUB_SCHEMA

    def test_assistant_string_content_no_crash(self):
        """Assistant message with string content (no tool_use blocks)."""
        trimmer = SystemPromptTrimmer()
        messages = [
            {"role": "assistant", "content": "Just text, no blocks"},
        ]
        body = _body(tools=[_tool_def("Read")], messages=messages)
        result = trimmer.trim(body)
        assert result.tools.stubbed_tools == 1

    def test_content_hash_deterministic(self):
        h1 = _content_hash("hello world")
        h2 = _content_hash("hello world")
        h3 = _content_hash("different")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16
