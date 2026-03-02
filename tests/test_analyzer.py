"""Tests for pichay.analyzer — system prompt waste analysis on proxy logs."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from pichay.analyzer import (
    Component,
    ComponentTrack,
    DuplicateGroup,
    SessionAnalysis,
    ToolUsageReport,
    _classify_reminder,
    _collect_tool_uses,
    _extract_message_injections,
    _extract_skill_entries,
    _extract_sp_components,
    _find_duplicate_skills,
    _infer_tool_definition_bytes,
    _split_into_sections,
    analyze_system_prompts,
    main,
    print_analysis,
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


# ---------------------------------------------------------------------------
# Realistic system prompt text (abridged to test section splitting)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEXT = """\
You are a Claude agent, built on Anthropic's Claude Agent SDK.
You are an interactive agent that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

# System
 - All text you output outside of tool use is displayed to the user.
 - Tools are executed in a user-selected permission mode.
 - Tool results and user messages may include <system-reminder> or other tags.

# Doing tasks
 - The user will primarily request you to perform software engineering tasks.
 - You are highly capable and often allow users to complete ambitious tasks.

# Executing actions with care
Carefully consider the reversibility and blast radius of actions.

# Using your tools
 - Do NOT use the Bash to run commands when a relevant dedicated tool is provided.
 - Break down and manage your work with the TodoWrite tool.

# Tone and style
 - Only use emojis if the user explicitly requests it.
 - Your responses should be short and concise.

# auto memory
You have a persistent auto memory directory at /home/user/.claude/memory/.

# Environment
You have been invoked in the following environment:
 - Primary working directory: /home/user/project
 - Platform: linux
 - Shell: bash

gitStatus: This is the git status at the start of the conversation.
Current branch: main
Status: clean

<fast_mode_info>
Fast mode uses the same model with faster output.
</fast_mode_info>
"""

SKILLS_REMINDER = """\
The following skills are available for use with the Skill tool:

- keybindings-help: Use when the user wants to customize keyboard shortcuts.
- pptx: Presentation creation, editing, and analysis.
- pdf: Comprehensive PDF manipulation toolkit.
- algorithmic-art: Creating algorithmic art using p5.js.
- example-skills:pptx: Presentation creation, editing, and analysis.
- example-skills:pdf: Comprehensive PDF manipulation toolkit.
- example-skills:algorithmic-art: Creating algorithmic art using p5.js.
- document-skills:pptx: Presentation creation, editing, and analysis.
- document-skills:pdf: Comprehensive PDF manipulation toolkit.
"""

BUDGET_REMINDER = "USD budget: $0.50/$20; $19.50 remaining"

CLAUDE_MD_REMINDER = """\
# claudeMd
Codebase and user instructions are shown below.

Contents of /home/user/project/CLAUDE.md:

# My Project
Run tests with: uv run pytest
"""

CURRENT_DATE_REMINDER = "# currentDate\nToday's date is 2026-03-01."


def _make_system_blocks(text: str = SYSTEM_PROMPT_TEXT) -> list[dict]:
    """Build a system prompt block list like the proxy captures."""
    return [{"type": "text", "text": text}]


def _make_user_msg_with_reminders(
    *reminder_texts: str,
    user_text: str = "Hello",
) -> dict:
    """Build a user message with system-reminder injections."""
    content = user_text
    for rt in reminder_texts:
        content += f"\n<system-reminder>\n{rt}\n</system-reminder>"
    return {"role": "user", "content": content}


def _make_user_msg_with_block_reminders(
    *reminder_texts: str,
    user_text: str = "Hello",
) -> dict:
    """Build a user message with block-style content including reminders."""
    blocks: list[dict] = [{"type": "text", "text": user_text}]
    for rt in reminder_texts:
        blocks.append({
            "type": "text",
            "text": f"<system-reminder>\n{rt}\n</system-reminder>",
        })
    return {"role": "user", "content": blocks}


def _tool_use(tid: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _tool_result(tid: str, content: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tid, "content": content}


def _make_request(
    system_prompt_full: list[dict] | None = None,
    messages_full: list[dict] | None = None,
    total_request_bytes: int = 50000,
    system_prompt_bytes: int = 15000,
    messages_total_bytes: int = 20000,
    ts: str = "2026-03-01T12:00:00Z",
) -> dict:
    """Build a proxy request record with all fields the analyzer expects."""
    rec: dict = {
        "type": "request",
        "timestamp": ts,
        "model": "claude-opus-4-6",
        "total_request_bytes": total_request_bytes,
        "system": {"system_prompt_bytes": system_prompt_bytes},
        "messages": {"messages_total_bytes": messages_total_bytes},
    }
    if system_prompt_full is not None:
        rec["system_prompt_full"] = system_prompt_full
    if messages_full is not None:
        rec["messages_full"] = messages_full
    return rec


# ---------------------------------------------------------------------------
# (1) Empty log file
# ---------------------------------------------------------------------------

class TestEmptyLog:
    def test_returns_zero_api_calls(self, tmp_path):
        p = _write_jsonl(tmp_path, "empty.jsonl", [])
        a = analyze_system_prompts(p)
        assert a.api_calls == 0
        assert a.total_system_prompt_bytes == 0
        assert a.total_message_injection_bytes == 0
        assert a.total_tool_definition_bytes == 0
        assert a.total_request_bytes == 0

    def test_properties_safe_on_empty(self, tmp_path):
        p = _write_jsonl(tmp_path, "empty.jsonl", [])
        a = analyze_system_prompts(p)
        assert a.static_pct == 0.0
        assert a.total_overhead_bytes == 0
        assert a.unused_tool_bytes == 0

    def test_empty_lists(self, tmp_path):
        p = _write_jsonl(tmp_path, "empty.jsonl", [])
        a = analyze_system_prompts(p)
        assert a.component_tracks == []
        assert a.sample_decomposition == []
        assert a.duplicate_groups == []

    def test_proxy_log_path_stored(self, tmp_path):
        p = _write_jsonl(tmp_path, "empty.jsonl", [])
        a = analyze_system_prompts(p)
        assert a.proxy_log == str(p)

    def test_non_request_records_ignored(self, tmp_path):
        """Response and compaction records are not counted as API calls."""
        records = [
            {"type": "response", "timestamp": "T1", "usage": {}},
            {"type": "compaction", "timestamp": "T2"},
            {"type": "page_faults", "timestamp": "T3"},
        ]
        p = _write_jsonl(tmp_path, "misc.jsonl", records)
        a = analyze_system_prompts(p)
        assert a.api_calls == 0


# ---------------------------------------------------------------------------
# (2) Log with no system_prompt_full field
# ---------------------------------------------------------------------------

class TestNoSystemPromptFull:
    def test_request_without_system_prompt_full(self, tmp_path):
        """Request records missing system_prompt_full produce zero SP bytes."""
        rec = {
            "type": "request",
            "timestamp": "2026-03-01T12:00:00Z",
            "model": "claude-opus-4-6",
            "total_request_bytes": 30000,
            "system": {"system_prompt_bytes": 10000},
            "messages": {"messages_total_bytes": 15000},
            "messages_full": [
                {"role": "user", "content": "hello"},
            ],
        }
        p = _write_jsonl(tmp_path, "no_sp.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert a.api_calls == 1
        assert a.total_system_prompt_bytes == 0
        # No SP components in sample decomposition
        sp_items = [
            d for d in a.sample_decomposition if d["source"] == "system_prompt"
        ]
        assert sp_items == []

    def test_system_prompt_full_empty_list(self, tmp_path):
        """system_prompt_full=[] produces zero SP bytes."""
        rec = _make_request(
            system_prompt_full=[],
            messages_full=[{"role": "user", "content": "hi"}],
        )
        p = _write_jsonl(tmp_path, "empty_sp.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert a.api_calls == 1
        assert a.total_system_prompt_bytes == 0

    def test_system_prompt_full_block_no_text(self, tmp_path):
        """Blocks without a 'text' key are skipped."""
        rec = _make_request(
            system_prompt_full=[{"type": "text"}],  # missing 'text'
            messages_full=[{"role": "user", "content": "hi"}],
        )
        p = _write_jsonl(tmp_path, "no_text.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert a.total_system_prompt_bytes == 0

    def test_system_prompt_full_non_dict_block(self, tmp_path):
        """Non-dict blocks in system_prompt_full are skipped."""
        rec = _make_request(
            system_prompt_full=["just a string", 42],
            messages_full=[{"role": "user", "content": "hi"}],
        )
        p = _write_jsonl(tmp_path, "non_dict.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert a.total_system_prompt_bytes == 0

    def test_messages_with_reminders_still_extracted(self, tmp_path):
        """Even without system_prompt_full, message injections are extracted."""
        rec = {
            "type": "request",
            "timestamp": "2026-03-01T12:00:00Z",
            "total_request_bytes": 10000,
            "system": {"system_prompt_bytes": 0},
            "messages": {"messages_total_bytes": 5000},
            "messages_full": [
                _make_user_msg_with_reminders(BUDGET_REMINDER),
            ],
        }
        p = _write_jsonl(tmp_path, "msg_only.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert a.total_message_injection_bytes > 0
        inj_items = [
            d for d in a.sample_decomposition
            if d["source"] == "message_injection"
        ]
        assert len(inj_items) == 1
        assert inj_items[0]["name"] == "budget_reminder"


# ---------------------------------------------------------------------------
# (3) Single-turn system prompt decomposition
# ---------------------------------------------------------------------------

class TestSingleTurnDecomposition:
    def test_section_splitting(self):
        """SYSTEM_PROMPT_TEXT is split into the expected sections."""
        blocks = _make_system_blocks()
        components = _extract_sp_components(blocks)
        names = [c.name for c in components]

        # Expected sections from the test prompt
        assert "agent_identity" in names
        assert "conversation_instructions" in names
        assert "system_section" in names
        assert "doing_tasks" in names
        assert "executing_actions" in names
        assert "using_tools" in names
        assert "tone_and_style" in names
        assert "auto_memory" in names
        assert "environment" in names
        assert "git_status" in names
        assert "fast_mode" in names

    def test_bytes_are_positive(self):
        """Every extracted component has positive byte count."""
        blocks = _make_system_blocks()
        components = _extract_sp_components(blocks)
        assert all(c.bytes > 0 for c in components)

    def test_content_hashes_populated(self):
        """Each component has a non-empty 16-char hex content hash."""
        blocks = _make_system_blocks()
        components = _extract_sp_components(blocks)
        for c in components:
            assert len(c.content_hash) == 16
            # hex chars only
            int(c.content_hash, 16)

    def test_same_text_same_hash(self):
        """Two components from identical text produce the same hash."""
        c1 = Component(
            name="a", source="system_prompt", text="hello", bytes=5,
        )
        c2 = Component(
            name="b", source="system_prompt", text="hello", bytes=5,
        )
        assert c1.content_hash == c2.content_hash

    def test_different_text_different_hash(self):
        c1 = Component(
            name="a", source="system_prompt", text="hello", bytes=5,
        )
        c2 = Component(
            name="b", source="system_prompt", text="world", bytes=5,
        )
        assert c1.content_hash != c2.content_hash

    def test_unsplittable_block_becomes_system_block_N(self):
        """A block with no recognized headings gets name system_block_<idx>."""
        blocks = [{"type": "text", "text": "Some random text with no headings."}]
        components = _extract_sp_components(blocks)
        assert len(components) == 1
        assert components[0].name == "system_block_0"

    def test_preamble_captured(self):
        """Text before the first recognized heading is captured as 'preamble'."""
        text = "Some preamble text.\n\n# System\nSystem content."
        sections = _split_into_sections(text)
        names = [n for n, _ in sections]
        assert "preamble" in names
        assert "system_section" in names

    def test_message_injection_extraction(self):
        """system-reminder blocks in messages are extracted and classified."""
        messages = [
            _make_user_msg_with_reminders(
                SKILLS_REMINDER,
                BUDGET_REMINDER,
                CLAUDE_MD_REMINDER,
            ),
        ]
        components = _extract_message_injections(messages)
        names = [c.name for c in components]

        assert "skills_list" in names
        assert "budget_reminder" in names
        assert "claude_md" in names
        assert all(c.source == "message_injection" for c in components)

    def test_block_style_message_injection(self):
        """Injections work with block-style (list) message content too."""
        messages = [
            _make_user_msg_with_block_reminders(BUDGET_REMINDER),
        ]
        components = _extract_message_injections(messages)
        assert len(components) == 1
        assert components[0].name == "budget_reminder"

    def test_current_date_classified(self):
        components = _extract_message_injections([
            _make_user_msg_with_reminders(CURRENT_DATE_REMINDER),
        ])
        assert components[0].name == "current_date"

    def test_unknown_reminder_classified(self):
        """Unrecognized reminder content gets name 'unknown_reminder'."""
        components = _extract_message_injections([
            _make_user_msg_with_reminders("Some unrecognized content here"),
        ])
        assert components[0].name == "unknown_reminder"

    def test_empty_reminder_skipped(self):
        """Empty system-reminder blocks produce no components."""
        msg = {"role": "user", "content": "<system-reminder>\n\n</system-reminder>"}
        components = _extract_message_injections([msg])
        assert components == []

    def test_full_turn_decomposition_in_analysis(self, tmp_path):
        """analyze_system_prompts populates sample_decomposition on first turn."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[
                _make_user_msg_with_reminders(SKILLS_REMINDER, BUDGET_REMINDER),
            ],
        )
        p = _write_jsonl(tmp_path, "decomp.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert len(a.sample_decomposition) > 0
        # Should have both sources
        sources = {d["source"] for d in a.sample_decomposition}
        assert "system_prompt" in sources
        assert "message_injection" in sources

        # Each entry has name, source, bytes
        for d in a.sample_decomposition:
            assert "name" in d
            assert "source" in d
            assert "bytes" in d
            assert d["bytes"] > 0

    def test_multiple_system_blocks(self):
        """Multiple system prompt blocks are each decomposed."""
        blocks = [
            {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
            {"type": "text", "text": "Some extra block with no known sections."},
        ]
        components = _extract_sp_components(blocks)
        assert len(components) == 2
        assert components[0].name == "agent_identity"
        assert components[1].name == "system_block_1"


# ---------------------------------------------------------------------------
# (4) Multi-turn static vs dynamic detection
# ---------------------------------------------------------------------------

class TestStaticVsDynamic:
    def _two_turn_log(self, tmp_path, sp_text_1, sp_text_2):
        """Create a 2-turn log with different system prompts."""
        rec1 = _make_request(
            system_prompt_full=_make_system_blocks(sp_text_1),
            messages_full=[{"role": "user", "content": "turn 1"}],
            ts="2026-03-01T12:00:00Z",
        )
        rec2 = _make_request(
            system_prompt_full=_make_system_blocks(sp_text_2),
            messages_full=[{"role": "user", "content": "turn 2"}],
            ts="2026-03-01T12:01:00Z",
        )
        return _write_jsonl(tmp_path, "multi.jsonl", [rec1, rec2])

    def test_identical_prompts_are_static(self, tmp_path):
        """When SP is identical across turns, all components are static."""
        p = self._two_turn_log(
            tmp_path, SYSTEM_PROMPT_TEXT, SYSTEM_PROMPT_TEXT,
        )
        a = analyze_system_prompts(p)

        assert a.api_calls == 2
        for track in a.component_tracks:
            assert track.is_static, f"{track.name} should be static"
            assert track.change_count == 0
            assert track.turns_present == 2
        assert a.dynamic_bytes == 0
        assert a.static_bytes > 0

    def test_changed_git_status_is_dynamic(self, tmp_path):
        """When git status changes between turns, that section is dynamic."""
        sp1 = SYSTEM_PROMPT_TEXT
        sp2 = SYSTEM_PROMPT_TEXT.replace(
            "Status: clean",
            "Status:\n?? newfile.py",
        )
        p = self._two_turn_log(tmp_path, sp1, sp2)
        a = analyze_system_prompts(p)

        track_by_name = {t.name: t for t in a.component_tracks}
        git_track = track_by_name.get("git_status")
        assert git_track is not None
        assert not git_track.is_static
        assert git_track.change_count == 1

        # Other sections should still be static
        for name in ["agent_identity", "system_section", "doing_tasks"]:
            t = track_by_name.get(name)
            if t:
                assert t.is_static, f"{name} should be static"

    def test_static_wasted_bytes(self, tmp_path):
        """Static re-send waste = sum of sizes[1:] for static components."""
        p = self._two_turn_log(
            tmp_path, SYSTEM_PROMPT_TEXT, SYSTEM_PROMPT_TEXT,
        )
        a = analyze_system_prompts(p)

        # Every component appears twice; the second send is wasted
        assert a.static_wasted_bytes > 0
        # Wasted should equal roughly the same as the first turn's SP bytes
        assert a.static_wasted_bytes == pytest.approx(
            a.total_system_prompt_bytes / 2, rel=0.01,
        )

    def test_dynamic_content_not_counted_as_wasted(self, tmp_path):
        """Changed content doesn't contribute to static_wasted_bytes."""
        sp1 = SYSTEM_PROMPT_TEXT
        sp2 = SYSTEM_PROMPT_TEXT.replace(
            "Status: clean",
            "Status:\n?? newfile.py",
        )
        p = self._two_turn_log(tmp_path, sp1, sp2)
        a = analyze_system_prompts(p)

        # dynamic_bytes should be nonzero (git_status appears twice, different)
        assert a.dynamic_bytes > 0
        # The static wasted should NOT include the git_status bytes
        track_by_name = {t.name: t for t in a.component_tracks}
        git_track = track_by_name["git_status"]
        assert git_track.wasted_bytes == 0

    def test_static_pct(self, tmp_path):
        """static_pct = static_bytes / (static + dynamic)."""
        p = self._two_turn_log(
            tmp_path, SYSTEM_PROMPT_TEXT, SYSTEM_PROMPT_TEXT,
        )
        a = analyze_system_prompts(p)

        # All static → 100%
        assert a.static_pct == pytest.approx(1.0)

    def test_component_tracks_sorted_by_total_bytes(self, tmp_path):
        """component_tracks are sorted descending by total_bytes_sent."""
        p = self._two_turn_log(
            tmp_path, SYSTEM_PROMPT_TEXT, SYSTEM_PROMPT_TEXT,
        )
        a = analyze_system_prompts(p)

        sizes = [t.total_bytes_sent for t in a.component_tracks]
        assert sizes == sorted(sizes, reverse=True)

    def test_three_turns_one_change(self, tmp_path):
        """3 turns: SP identical in turns 1&2, changes in turn 3."""
        sp_changed = SYSTEM_PROMPT_TEXT.replace("Platform: linux", "Platform: darwin")
        recs = [
            _make_request(
                system_prompt_full=_make_system_blocks(SYSTEM_PROMPT_TEXT),
                messages_full=[{"role": "user", "content": "t1"}],
                ts="2026-03-01T12:00:00Z",
            ),
            _make_request(
                system_prompt_full=_make_system_blocks(SYSTEM_PROMPT_TEXT),
                messages_full=[{"role": "user", "content": "t2"}],
                ts="2026-03-01T12:01:00Z",
            ),
            _make_request(
                system_prompt_full=_make_system_blocks(sp_changed),
                messages_full=[{"role": "user", "content": "t3"}],
                ts="2026-03-01T12:02:00Z",
            ),
        ]
        p = _write_jsonl(tmp_path, "three.jsonl", recs)
        a = analyze_system_prompts(p)

        track_by_name = {t.name: t for t in a.component_tracks}
        env_track = track_by_name.get("environment")
        assert env_track is not None
        assert env_track.turns_present == 3
        assert env_track.change_count == 1
        assert not env_track.is_static


# ---------------------------------------------------------------------------
# (5) Tool usage tracking with used and unused tools
# ---------------------------------------------------------------------------

class TestToolUsageTracking:
    def test_used_tools_counted(self, tmp_path):
        """Tools appearing in messages_full are counted correctly."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[
                {"role": "user", "content": "do something"},
                {"role": "assistant", "content": [
                    _tool_use("tu_1", "Read", {"file_path": "/foo.py"}),
                ]},
                {"role": "user", "content": [
                    _tool_result("tu_1", "file contents here"),
                ]},
                {"role": "assistant", "content": [
                    _tool_use("tu_2", "Grep", {"pattern": "TODO"}),
                ]},
                {"role": "user", "content": [
                    _tool_result("tu_2", "line 10: TODO fix this"),
                ]},
                {"role": "assistant", "content": [
                    _tool_use("tu_3", "Read", {"file_path": "/bar.py"}),
                ]},
                {"role": "user", "content": [
                    _tool_result("tu_3", "bar contents"),
                ]},
            ],
        )
        p = _write_jsonl(tmp_path, "tools.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert "Read" in a.tool_usage.used_tools
        assert "Grep" in a.tool_usage.used_tools
        assert a.tool_usage.tool_use_counts["Read"] == 2
        assert a.tool_usage.tool_use_counts["Grep"] == 1

    def test_unused_tools_identified(self, tmp_path):
        """Tools in _KNOWN_TOOLS but not used are listed as unused."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    _tool_use("tu_1", "Read", {"file_path": "/f.py"}),
                ]},
                {"role": "user", "content": [
                    _tool_result("tu_1", "data"),
                ]},
            ],
        )
        p = _write_jsonl(tmp_path, "unused.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert "Read" not in a.tool_usage.unused_tools
        assert "Bash" in a.tool_usage.unused_tools
        assert "Write" in a.tool_usage.unused_tools
        assert "WebFetch" in a.tool_usage.unused_tools

    def test_no_tool_usage_all_unused(self, tmp_path):
        """When no tools are used, all known tools are unused."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[{"role": "user", "content": "hello"}],
        )
        p = _write_jsonl(tmp_path, "no_tools.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert a.tool_usage.used_tools == []
        assert len(a.tool_usage.unused_tools) == len(a.tool_usage.defined_tools)

    def test_tool_definition_bytes_inferred(self, tmp_path):
        """Tool def bytes = gap between total_request and (system + messages)."""
        # total=50000, system=15000, messages=20000 → gap=15000
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[{"role": "user", "content": "hi"}],
            total_request_bytes=50000,
            system_prompt_bytes=15000,
            messages_total_bytes=20000,
        )
        p = _write_jsonl(tmp_path, "gap.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert a.tool_usage.tool_definition_bytes == 15000

    def test_tool_definition_bytes_median(self, tmp_path):
        """With multiple requests, the median gap is used."""
        recs = [
            _make_request(
                system_prompt_full=_make_system_blocks(),
                messages_full=[{"role": "user", "content": "t1"}],
                total_request_bytes=50000,
                system_prompt_bytes=15000,
                messages_total_bytes=20000,
                ts="2026-03-01T12:00:00Z",
            ),
            _make_request(
                system_prompt_full=_make_system_blocks(),
                messages_full=[{"role": "user", "content": "t2"}],
                total_request_bytes=52000,
                system_prompt_bytes=15000,
                messages_total_bytes=22000,
                ts="2026-03-01T12:01:00Z",
            ),
            _make_request(
                system_prompt_full=_make_system_blocks(),
                messages_full=[{"role": "user", "content": "t3"}],
                total_request_bytes=99999,  # outlier
                system_prompt_bytes=15000,
                messages_total_bytes=20000,
                ts="2026-03-01T12:02:00Z",
            ),
        ]
        p = _write_jsonl(tmp_path, "median.jsonl", recs)
        a = analyze_system_prompts(p)

        # Gaps: 15000, 15000, 64999 → sorted: [15000, 15000, 64999] → median idx 1 = 15000
        assert a.tool_usage.tool_definition_bytes == 15000

    def test_total_tool_definition_bytes_scaled(self, tmp_path):
        """total_tool_definition_bytes = per-request bytes * api_calls."""
        recs = [
            _make_request(
                system_prompt_full=_make_system_blocks(),
                messages_full=[{"role": "user", "content": f"t{i}"}],
                total_request_bytes=50000,
                system_prompt_bytes=15000,
                messages_total_bytes=20000,
                ts=f"2026-03-01T12:0{i}:00Z",
            )
            for i in range(3)
        ]
        p = _write_jsonl(tmp_path, "scaled.jsonl", recs)
        a = analyze_system_prompts(p)

        assert a.total_tool_definition_bytes == 15000 * 3

    def test_unused_tool_bytes_estimate(self, tmp_path):
        """unused_tool_bytes = (per_tool_bytes * n_unused)."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    _tool_use("tu_1", "Read", {"file_path": "/f.py"}),
                ]},
                {"role": "user", "content": [
                    _tool_result("tu_1", "data"),
                ]},
            ],
            total_request_bytes=50000,
            system_prompt_bytes=15000,
            messages_total_bytes=20000,
        )
        p = _write_jsonl(tmp_path, "est.jsonl", [rec])
        a = analyze_system_prompts(p)

        n_defined = len(a.tool_usage.defined_tools)
        n_unused = len(a.tool_usage.unused_tools)
        per_tool = a.tool_usage.tool_definition_bytes / n_defined
        expected = int(per_tool * n_unused)
        assert a.unused_tool_bytes == expected

    def test_collect_tool_uses_skips_non_request(self):
        """_collect_tool_uses only processes request records."""
        records = [
            {"type": "response", "messages_full": [
                {"role": "assistant", "content": [
                    _tool_use("tu_1", "Read", {}),
                ]},
            ]},
        ]
        counts = _collect_tool_uses(records)
        assert counts == {}

    def test_collect_tool_uses_string_content_skipped(self):
        """Messages with string content (not list) are safely skipped."""
        records = [
            {"type": "request", "messages_full": [
                {"role": "assistant", "content": "just text"},
            ]},
        ]
        counts = _collect_tool_uses(records)
        assert counts == {}


# ---------------------------------------------------------------------------
# (6) Duplicate skill detection
# ---------------------------------------------------------------------------

class TestDuplicateSkillDetection:
    def test_detects_duplicate_skills(self):
        """Skills with the same base name under different prefixes are dupes."""
        comp = Component(
            name="skills_list",
            source="message_injection",
            text=SKILLS_REMINDER,
            bytes=len(SKILLS_REMINDER.encode("utf-8")),
        )
        groups = _find_duplicate_skills([comp])

        # pptx appears as: pptx, example-skills:pptx, document-skills:pptx
        # pdf appears as: pdf, example-skills:pdf, document-skills:pdf
        # algorithmic-art: algorithmic-art, example-skills:algorithmic-art
        group_names = {g.canonical_name for g in groups}
        assert "pptx" in group_names
        assert "pdf" in group_names
        assert "algorithmic-art" in group_names

    def test_duplicate_group_members(self):
        """Each duplicate group lists all the full names."""
        comp = Component(
            name="skills_list",
            source="message_injection",
            text=SKILLS_REMINDER,
            bytes=len(SKILLS_REMINDER.encode("utf-8")),
        )
        groups = _find_duplicate_skills([comp])
        by_name = {g.canonical_name: g for g in groups}

        pptx_group = by_name["pptx"]
        assert "pptx" in pptx_group.members
        assert "example-skills:pptx" in pptx_group.members
        assert "document-skills:pptx" in pptx_group.members
        assert len(pptx_group.members) == 3

    def test_duplicate_bytes_calculated(self):
        """duplicate_bytes = total_bytes - instance_bytes."""
        comp = Component(
            name="skills_list",
            source="message_injection",
            text=SKILLS_REMINDER,
            bytes=len(SKILLS_REMINDER.encode("utf-8")),
        )
        groups = _find_duplicate_skills([comp])
        for g in groups:
            assert g.duplicate_bytes == g.total_bytes - g.instance_bytes
            assert g.duplicate_bytes > 0

    def test_no_duplicates_when_unique_skills(self):
        """Skills with unique base names produce no duplicate groups."""
        text = """\
- keybindings-help: Customize keyboard shortcuts.
- pptx: Presentation creation.
- pdf: PDF manipulation.
"""
        comp = Component(
            name="skills_list",
            source="message_injection",
            text=text,
            bytes=len(text.encode("utf-8")),
        )
        groups = _find_duplicate_skills([comp])
        assert groups == []

    def test_non_skills_components_ignored(self):
        """Only components named 'skills_list' are checked for duplicates."""
        comp = Component(
            name="budget_reminder",
            source="message_injection",
            text=SKILLS_REMINDER,  # same text, wrong name
            bytes=len(SKILLS_REMINDER.encode("utf-8")),
        )
        groups = _find_duplicate_skills([comp])
        assert groups == []

    def test_extract_skill_entries(self):
        """_extract_skill_entries parses skill names correctly."""
        entries = _extract_skill_entries(SKILLS_REMINDER)
        full_names = [e[0] for e in entries]
        base_names = [e[1] for e in entries]

        assert "keybindings-help" in full_names
        assert "example-skills:pptx" in full_names
        assert "pptx" in base_names
        assert "pdf" in base_names

    def test_extract_skill_entries_colon_split(self):
        """Base name is the part after the last colon."""
        entries = _extract_skill_entries(
            "- foo:bar:baz: Some description."
        )
        assert entries[0] == ("foo:bar:baz", "baz")

    def test_duplicate_detection_in_full_analysis(self, tmp_path):
        """analyze_system_prompts populates duplicate_groups from messages."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[
                _make_user_msg_with_reminders(SKILLS_REMINDER),
            ],
        )
        p = _write_jsonl(tmp_path, "dupes.jsonl", [rec])
        a = analyze_system_prompts(p)

        assert len(a.duplicate_groups) > 0
        assert a.duplicate_bytes > 0

    def test_duplicates_only_from_first_turn(self, tmp_path):
        """Duplicate detection uses only the first turn's message injections."""
        no_dupes_skills = """\
- unique-skill-a: Does A.
- unique-skill-b: Does B.
"""
        recs = [
            _make_request(
                system_prompt_full=_make_system_blocks(),
                messages_full=[
                    _make_user_msg_with_reminders(no_dupes_skills),
                ],
                ts="2026-03-01T12:00:00Z",
            ),
            _make_request(
                system_prompt_full=_make_system_blocks(),
                messages_full=[
                    _make_user_msg_with_reminders(SKILLS_REMINDER),
                ],
                ts="2026-03-01T12:01:00Z",
            ),
        ]
        p = _write_jsonl(tmp_path, "first_only.jsonl", recs)
        a = analyze_system_prompts(p)

        # Turn 1 has no dupes, so even though turn 2 has dupes, groups = []
        assert a.duplicate_groups == []
        assert a.duplicate_bytes == 0


# ---------------------------------------------------------------------------
# (7) Session summary percentage calculations
# ---------------------------------------------------------------------------

class TestSessionSummaryPercentages:
    def test_static_pct_all_static(self):
        """100% static when all content is identical across turns."""
        a = SessionAnalysis(proxy_log="x", static_bytes=1000, dynamic_bytes=0)
        assert a.static_pct == pytest.approx(1.0)

    def test_static_pct_all_dynamic(self):
        a = SessionAnalysis(proxy_log="x", static_bytes=0, dynamic_bytes=1000)
        assert a.static_pct == pytest.approx(0.0)

    def test_static_pct_mixed(self):
        a = SessionAnalysis(proxy_log="x", static_bytes=750, dynamic_bytes=250)
        assert a.static_pct == pytest.approx(0.75)

    def test_static_pct_zero_zero(self):
        a = SessionAnalysis(proxy_log="x", static_bytes=0, dynamic_bytes=0)
        assert a.static_pct == 0.0

    def test_total_overhead_bytes(self):
        a = SessionAnalysis(
            proxy_log="x",
            total_system_prompt_bytes=10000,
            total_message_injection_bytes=5000,
            total_tool_definition_bytes=8000,
        )
        assert a.total_overhead_bytes == 23000

    def test_unused_tool_bytes_with_data(self):
        a = SessionAnalysis(
            proxy_log="x",
            tool_usage=ToolUsageReport(
                defined_tools=["A", "B", "C", "D"],
                used_tools=["A"],
                unused_tools=["B", "C", "D"],
                tool_definition_bytes=4000,
            ),
        )
        # per_tool = 4000/4 = 1000, unused = 3 → 3000
        assert a.unused_tool_bytes == 3000

    def test_unused_tool_bytes_no_tools(self):
        a = SessionAnalysis(proxy_log="x")
        assert a.unused_tool_bytes == 0

    def test_unused_tool_bytes_all_used(self):
        a = SessionAnalysis(
            proxy_log="x",
            tool_usage=ToolUsageReport(
                defined_tools=["A", "B"],
                used_tools=["A", "B"],
                unused_tools=[],
                tool_definition_bytes=2000,
            ),
        )
        assert a.unused_tool_bytes == 0

    def test_component_track_properties(self):
        """ComponentTrack computed properties are correct."""
        track = ComponentTrack(
            name="test",
            hashes=["aaa", "aaa", "aaa"],
            sizes=[100, 100, 100],
        )
        assert track.is_static is True
        assert track.total_bytes_sent == 300
        assert track.unique_bytes == 100
        assert track.wasted_bytes == 200
        assert track.turns_present == 3
        assert track.change_count == 0

    def test_component_track_dynamic(self):
        track = ComponentTrack(
            name="test",
            hashes=["aaa", "bbb", "aaa"],
            sizes=[100, 120, 100],
        )
        assert track.is_static is False
        assert track.wasted_bytes == 0  # not static → no wasted
        assert track.change_count == 2  # aaa→bbb, bbb→aaa

    def test_component_track_empty(self):
        track = ComponentTrack(name="empty")
        assert track.is_static is True  # empty set has <= 1 unique
        assert track.total_bytes_sent == 0
        assert track.unique_bytes == 0
        assert track.wasted_bytes == 0
        assert track.turns_present == 0
        assert track.change_count == 0

    def test_component_track_single_turn(self):
        track = ComponentTrack(
            name="once", hashes=["abc"], sizes=[500],
        )
        assert track.is_static is True
        assert track.wasted_bytes == 0  # only 1 send, nothing wasted
        assert track.unique_bytes == 500

    def test_duplicate_group_properties(self):
        g = DuplicateGroup(
            canonical_name="pdf",
            members=["pdf", "example-skills:pdf", "document-skills:pdf"],
            instance_bytes=200,
            total_bytes=600,
        )
        assert g.duplicate_bytes == 400

    def test_duplicate_group_single_member(self):
        """A group with one member has no duplicate bytes."""
        g = DuplicateGroup(
            canonical_name="unique",
            members=["unique"],
            instance_bytes=200,
            total_bytes=200,
        )
        assert g.duplicate_bytes == 0

    def test_full_session_summary_integration(self, tmp_path):
        """End-to-end: check that summary fields are populated correctly."""
        recs = [
            _make_request(
                system_prompt_full=_make_system_blocks(),
                messages_full=[
                    _make_user_msg_with_reminders(SKILLS_REMINDER, BUDGET_REMINDER),
                    {"role": "assistant", "content": [
                        _tool_use("tu_1", "Read", {"file_path": "/f.py"}),
                    ]},
                    {"role": "user", "content": [
                        _tool_result("tu_1", "file data"),
                    ]},
                ],
                total_request_bytes=50000,
                system_prompt_bytes=15000,
                messages_total_bytes=20000,
                ts="2026-03-01T12:00:00Z",
            ),
            _make_request(
                system_prompt_full=_make_system_blocks(),
                messages_full=[
                    _make_user_msg_with_reminders(SKILLS_REMINDER, BUDGET_REMINDER),
                    {"role": "assistant", "content": [
                        _tool_use("tu_2", "Bash", {"command": "ls"}),
                    ]},
                    {"role": "user", "content": [
                        _tool_result("tu_2", "file1\nfile2"),
                    ]},
                ],
                total_request_bytes=55000,
                system_prompt_bytes=15000,
                messages_total_bytes=25000,
                ts="2026-03-01T12:01:00Z",
            ),
        ]
        p = _write_jsonl(tmp_path, "full.jsonl", recs)
        a = analyze_system_prompts(p)

        assert a.api_calls == 2
        assert a.total_system_prompt_bytes > 0
        assert a.total_message_injection_bytes > 0
        assert a.total_tool_definition_bytes > 0
        assert a.total_request_bytes == 50000 + 55000
        assert a.total_overhead_bytes == (
            a.total_system_prompt_bytes
            + a.total_message_injection_bytes
            + a.total_tool_definition_bytes
        )
        assert "Read" in a.tool_usage.used_tools
        assert "Bash" in a.tool_usage.used_tools
        assert a.tool_usage.tool_use_counts["Read"] == 1
        assert a.tool_usage.tool_use_counts["Bash"] == 1


# ---------------------------------------------------------------------------
# (8) CLI entry point with --json flag
# ---------------------------------------------------------------------------

class TestCLIEntryPoint:
    def test_json_output(self, tmp_path, capsys, monkeypatch):
        """--json outputs valid JSON with expected fields."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[
                _make_user_msg_with_reminders(SKILLS_REMINDER, BUDGET_REMINDER),
                {"role": "assistant", "content": [
                    _tool_use("tu_1", "Read", {"file_path": "/f.py"}),
                ]},
                {"role": "user", "content": [
                    _tool_result("tu_1", "data"),
                ]},
            ],
        )
        p = _write_jsonl(tmp_path, "cli.jsonl", [rec])

        monkeypatch.setattr(sys, "argv", ["analyzer", "--json", str(p)])
        main()

        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert data["proxy_log"] == str(p)
        assert data["api_calls"] == 1
        assert "total_system_prompt_bytes" in data
        assert "total_message_injection_bytes" in data
        assert "total_tool_definition_bytes" in data
        assert "tool_usage" in data
        assert "duplicate_groups" in data

    def test_json_trims_sample_decomposition(self, tmp_path, capsys, monkeypatch):
        """--json removes sample_decomposition from output."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[{"role": "user", "content": "hi"}],
        )
        p = _write_jsonl(tmp_path, "trim.jsonl", [rec])

        monkeypatch.setattr(sys, "argv", ["analyzer", "--json", str(p)])
        main()

        data = json.loads(capsys.readouterr().out.strip())
        assert "sample_decomposition" not in data

    def test_json_component_tracks_flattened(self, tmp_path, capsys, monkeypatch):
        """--json flattens component_tracks to summary dicts."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[{"role": "user", "content": "hi"}],
        )
        p = _write_jsonl(tmp_path, "flat.jsonl", [rec])

        monkeypatch.setattr(sys, "argv", ["analyzer", "--json", str(p)])
        main()

        data = json.loads(capsys.readouterr().out.strip())
        tracks = data["component_tracks"]
        assert isinstance(tracks, list)
        for t in tracks:
            assert "name" in t
            assert "total_bytes_sent" in t
            assert "turns_present" in t
            assert "is_static" in t
            assert "change_count" in t
            # Should NOT have raw hashes/sizes
            assert "hashes" not in t
            assert "sizes" not in t

    def test_human_readable_output(self, tmp_path, capsys, monkeypatch):
        """Without --json, output contains human-readable sections."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[
                _make_user_msg_with_reminders(SKILLS_REMINDER),
                {"role": "assistant", "content": [
                    _tool_use("tu_1", "Read", {"file_path": "/f.py"}),
                ]},
                {"role": "user", "content": [
                    _tool_result("tu_1", "data"),
                ]},
            ],
        )
        p = _write_jsonl(tmp_path, "human.jsonl", [rec])

        monkeypatch.setattr(sys, "argv", ["analyzer", str(p)])
        main()

        out = capsys.readouterr().out
        assert "SYSTEM PROMPT WASTE ANALYSIS" in out
        assert "Component Decomposition" in out
        assert "Static vs Dynamic" in out
        assert "Tool Usage" in out
        assert "Session Summary" in out

    def test_human_readable_shows_duplicates(self, tmp_path, capsys, monkeypatch):
        """Human-readable output includes duplicate section when dupes exist."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[
                _make_user_msg_with_reminders(SKILLS_REMINDER),
            ],
        )
        p = _write_jsonl(tmp_path, "dupes_hr.jsonl", [rec])

        monkeypatch.setattr(sys, "argv", ["analyzer", str(p)])
        main()

        out = capsys.readouterr().out
        assert "Duplicate Content" in out

    def test_missing_file_exits(self, tmp_path, monkeypatch):
        """Non-existent log file produces error and sys.exit(1)."""
        fake = tmp_path / "does_not_exist.jsonl"
        monkeypatch.setattr(sys, "argv", ["analyzer", str(fake)])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_print_analysis_no_crash(self, tmp_path, capsys):
        """print_analysis runs without error on a real analysis."""
        rec = _make_request(
            system_prompt_full=_make_system_blocks(),
            messages_full=[
                _make_user_msg_with_reminders(
                    SKILLS_REMINDER, BUDGET_REMINDER, CLAUDE_MD_REMINDER,
                ),
                {"role": "assistant", "content": [
                    _tool_use("tu_1", "Read", {"file_path": "/f.py"}),
                ]},
                {"role": "user", "content": [
                    _tool_result("tu_1", "data"),
                ]},
            ],
        )
        p = _write_jsonl(tmp_path, "pa.jsonl", [rec])
        a = analyze_system_prompts(p)
        print_analysis(a)
        out = capsys.readouterr().out
        assert "SYSTEM PROMPT WASTE ANALYSIS" in out

    def test_print_analysis_empty_session(self, capsys):
        """print_analysis on an empty SessionAnalysis doesn't crash."""
        a = SessionAnalysis(proxy_log="empty.jsonl")
        print_analysis(a)
        out = capsys.readouterr().out
        assert "SYSTEM PROMPT WASTE ANALYSIS" in out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_classify_reminder_memory_md(self):
        assert _classify_reminder("# memoryMd\nSome memory content") == "memory_md"
        assert _classify_reminder("Your MEMORY.md file says...") == "memory_md"

    def test_classify_reminder_todo(self):
        assert _classify_reminder(
            "The TodoWrite tool hasn't been used yet"
        ) == "todo_reminder"

    def test_classify_reminder_unknown(self):
        assert _classify_reminder("Something completely unrecognized") == "unknown_reminder"

    def test_infer_tool_definition_bytes_no_gap(self):
        """When total == system + messages, gap is 0 → returns 0."""
        records = [
            {
                "type": "request",
                "total_request_bytes": 100,
                "system": {"system_prompt_bytes": 50},
                "messages": {"messages_total_bytes": 50},
            },
        ]
        assert _infer_tool_definition_bytes(records) == 0

    def test_infer_tool_definition_bytes_negative_gap(self):
        """Negative gap (shouldn't happen in practice) is filtered out."""
        records = [
            {
                "type": "request",
                "total_request_bytes": 10,
                "system": {"system_prompt_bytes": 50},
                "messages": {"messages_total_bytes": 50},
            },
        ]
        # gap = 10 - 50 - 50 = -90, not > 0, so filtered → returns 0
        assert _infer_tool_definition_bytes(records) == 0

    def test_session_analysis_asdict(self):
        """SessionAnalysis can be serialized via asdict without error."""
        a = SessionAnalysis(
            proxy_log="test.jsonl",
            api_calls=5,
            total_system_prompt_bytes=10000,
        )
        d = asdict(a)
        assert d["proxy_log"] == "test.jsonl"
        assert d["api_calls"] == 5
        # Ensure JSON-serializable
        json.dumps(d)

    def test_split_into_sections_no_matches(self):
        """Text with no recognized headings returns empty list."""
        assert _split_into_sections("Just some random text.") == []

    def test_split_into_sections_single_heading(self):
        """Text with one recognized heading returns that section."""
        text = "# System\nSome system content."
        sections = _split_into_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "system_section"

    def test_message_with_non_list_non_string_content(self):
        """Messages with unexpected content types are safely skipped."""
        messages = [
            {"role": "user", "content": 12345},
        ]
        components = _extract_message_injections(messages)
        assert components == []

    def test_message_block_non_string_text(self):
        """Content blocks where 'text' is not a string are skipped."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": 42},
            ]},
        ]
        components = _extract_message_injections(messages)
        assert components == []
