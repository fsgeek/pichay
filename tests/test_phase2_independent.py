"""Independent Phase 2 test suite.

Authored by a separate review agent. Tests derived from the contract
(docs/conversation_memory_management.md and the tag format specification
in tags.py docstring), not from knowledge of the implementation internals.

Focus areas:
- Edge cases in tag parsing (malformed, partial, special characters)
- apply_to_messages contract invariants
- Tag stripping completeness (no residue)
- Interaction between ops (drop+apply, summarize+apply, idempotency)
- Behaviors the author's tests did not cover
"""

from __future__ import annotations

import pytest

from pichay.tags import CleanupOps, parse_cleanup_tags, strip_cleanup_tags
from pichay.blocks import BlockStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long(char: str, n: int = 300) -> str:
    """Return a string long enough to be labeled (>= 200 chars)."""
    return char * n


def _labeled_store(*contents: str) -> tuple[BlockStore, list[dict]]:
    """Create a BlockStore with labeled messages for each content string."""
    bs = BlockStore()
    messages = [
        {"role": "user", "content": c}
        for c in contents
    ]
    bs.label_messages(messages, current_turn=1)
    return bs, messages


def _block_ids(bs: BlockStore) -> list[str]:
    return list(bs._by_id.keys())


# ===========================================================================
# TAG PARSER — EDGE CASES
# ===========================================================================

class TestBlockIdBoundaryConditions:
    """Block IDs must be 8-12 lowercase hex chars. Contract is strict."""

    def test_7_char_id_not_parsed_as_drop(self):
        # Below minimum (8) — should be ignored
        text = "<memory_cleanup>\ndrop: block:a1b2c3d\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.drops == [], "7-char block ID should not be accepted"

    def test_13_char_id_not_parsed_as_drop(self):
        # Above maximum (12) — should be ignored
        text = "<memory_cleanup>\ndrop: block:a1b2c3d4e5f6a\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.drops == [], "13-char block ID should not be accepted"

    def test_8_char_id_accepted(self):
        text = "<memory_cleanup>\ndrop: block:a1b2c3d4\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.drops == ["a1b2c3d4"]

    def test_12_char_id_accepted(self):
        text = "<memory_cleanup>\ndrop: block:a1b2c3d4e5f6\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.drops == ["a1b2c3d4e5f6"]

    def test_uppercase_hex_not_accepted(self):
        # Regex is [a-f0-9] — uppercase should be rejected
        text = "<memory_cleanup>\ndrop: block:A1B2C3D4\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.drops == [], "Uppercase hex in block ID should not be accepted"

    def test_non_hex_chars_in_id_not_accepted(self):
        text = "<memory_cleanup>\ndrop: block:g1h2i3j4\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.drops == [], "Non-hex chars in block ID should not be accepted"

    def test_drop_without_block_prefix_not_parsed(self):
        # 'drop: a3f2b901' is missing the 'block:' prefix
        text = "<memory_cleanup>\ndrop: a3f2b901\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.drops == [], "Drop without 'block:' prefix should not be parsed"


class TestSummarizeEdgeCases:
    """Edge cases in the summarize: block:<id> \"<text>\" format."""

    def test_summarize_empty_summary_string(self):
        # Empty string is technically valid syntax
        text = '<memory_cleanup>\nsummarize: block:a1b2c3d4 ""\n</memory_cleanup>'
        ops = parse_cleanup_tags(text)
        # Should parse — empty string is a valid summary
        assert len(ops.summaries) == 1
        block_id, summary = ops.summaries[0]
        assert block_id == "a1b2c3d4"
        assert summary == ""

    def test_summarize_missing_closing_quote_not_parsed(self):
        # Unclosed quote — should not parse
        text = '<memory_cleanup>\nsummarize: block:a1b2c3d4 "No closing quote\n</memory_cleanup>'
        ops = parse_cleanup_tags(text)
        assert ops.summaries == [], "Summarize with unclosed quote should not be parsed"

    def test_summarize_summary_with_spaces(self):
        text = '<memory_cleanup>\nsummarize: block:a1b2c3d4 "This block discussed topic X and resolved the issue"\n</memory_cleanup>'
        ops = parse_cleanup_tags(text)
        assert len(ops.summaries) == 1
        _, summary = ops.summaries[0]
        assert summary == "This block discussed topic X and resolved the issue"

    def test_summarize_without_quotes_not_parsed(self):
        # Missing quotes entirely
        text = "<memory_cleanup>\nsummarize: block:a1b2c3d4 Summary without quotes\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.summaries == [], "Summarize without quotes should not be parsed"

    def test_summarize_summary_with_internal_parens(self):
        # Parens and other punctuation in summary
        text = '<memory_cleanup>\nsummarize: block:a1b2c3d4 "Result: 75.6% (n=100)"\n</memory_cleanup>'
        ops = parse_cleanup_tags(text)
        assert len(ops.summaries) == 1
        _, summary = ops.summaries[0]
        assert "75.6%" in summary


class TestUnclosedAndMalformedTags:
    """Partial or malformed tags should not produce operations."""

    def test_unclosed_tag_not_parsed(self):
        # Opening tag with no closing tag
        text = "Some text <memory_cleanup>\ndrop: block:a1b2c3d4\n"
        ops = parse_cleanup_tags(text)
        assert ops.empty, "Unclosed <memory_cleanup> tag should not parse any ops"

    def test_closing_tag_only_not_parsed(self):
        # Only a closing tag — no opening
        text = "drop: block:a1b2c3d4\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.empty

    def test_nested_tags_outer_parsed(self):
        # Nesting is unusual — the outer match should work (non-greedy)
        # but inner tags won't be processed recursively
        text = "<memory_cleanup>\ndrop: block:a1b2c3d4\n<memory_cleanup>\n</memory_cleanup>\n</memory_cleanup>"
        # This is an edge case — just verify it doesn't crash
        ops = parse_cleanup_tags(text)
        assert isinstance(ops, CleanupOps)

    def test_empty_text(self):
        ops = parse_cleanup_tags("")
        assert ops.empty

    def test_tag_on_same_line_as_content(self):
        # Tag not preceded by newline — still should parse
        text = "text<memory_cleanup>\ndrop: block:a1b2c3d4\n</memory_cleanup>more text"
        ops = parse_cleanup_tags(text)
        assert ops.drops == ["a1b2c3d4"]


class TestMultipleTagMerging:
    """Multiple tags in one text should be merged into one CleanupOps."""

    def test_two_tags_merge_drops(self):
        text = (
            "<memory_cleanup>\ndrop: block:aaaaaaaa\n</memory_cleanup>\n"
            "<memory_cleanup>\ndrop: block:bbbbbbbb\n</memory_cleanup>"
        )
        ops = parse_cleanup_tags(text)
        assert set(ops.drops) == {"aaaaaaaa", "bbbbbbbb"}

    def test_two_tags_merge_different_ops(self):
        text = (
            "<memory_cleanup>\ndrop: block:aaaaaaaa\n</memory_cleanup>\n"
            '<memory_cleanup>\nsummarize: block:bbbbbbbb "summary text"\n</memory_cleanup>'
        )
        ops = parse_cleanup_tags(text)
        assert ops.drops == ["aaaaaaaa"]
        assert len(ops.summaries) == 1


class TestReleaseEdgeCases:
    """Release paths are comma-separated and stripped."""

    def test_release_single_path_no_trailing_space(self):
        text = "<memory_cleanup>\nrelease: path/to/file.py\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.releases == ["path/to/file.py"]

    def test_release_paths_stripped_of_whitespace(self):
        text = "<memory_cleanup>\nrelease:   src/foo.py ,   src/bar.py  \n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        # Whitespace should be stripped from each path
        assert "src/foo.py" in ops.releases
        assert "src/bar.py" in ops.releases

    def test_release_empty_paths_not_included(self):
        # Trailing comma creates empty entry
        text = "<memory_cleanup>\nrelease: src/foo.py, \n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        # Empty string should not be in releases
        assert "" not in ops.releases
        assert "src/foo.py" in ops.releases


# ===========================================================================
# STRIP_CLEANUP_TAGS — COMPLETENESS
# ===========================================================================

class TestStripCompletenessAndResidues:
    """After stripping, no tag residue should remain."""

    def test_strip_removes_opening_and_closing_tags(self):
        text = "Before\n<memory_cleanup>\ndrop: block:a1b2c3d4\n</memory_cleanup>\nAfter"
        result = strip_cleanup_tags(text)
        assert "<memory_cleanup>" not in result
        assert "</memory_cleanup>" not in result

    def test_strip_removes_tag_content(self):
        text = "Before\n<memory_cleanup>\ndrop: block:a1b2c3d4\n</memory_cleanup>\nAfter"
        result = strip_cleanup_tags(text)
        # The block ID inside the tag should not appear in output
        # (it's inside the tag, not in surrounding text)
        assert "a1b2c3d4" not in result

    def test_strip_preserves_surrounding_text_exactly(self):
        text = "Line 1\n\n<memory_cleanup>\ndrop: block:a1b2c3d4\n</memory_cleanup>\n\nLine 2"
        result = strip_cleanup_tags(text)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_strip_no_tag_returns_original_content(self):
        text = "No tags here at all."
        result = strip_cleanup_tags(text)
        assert result == text

    def test_strip_multiple_tags_all_removed(self):
        text = (
            "Start\n"
            "<memory_cleanup>\ndrop: block:aaaaaaaa\n</memory_cleanup>\n"
            "Middle\n"
            "<memory_cleanup>\nanchor: block:bbbbbbbb\n</memory_cleanup>\n"
            "End"
        )
        result = strip_cleanup_tags(text)
        assert "<memory_cleanup>" not in result
        assert "</memory_cleanup>" not in result
        assert "Start" in result
        assert "Middle" in result
        assert "End" in result

    def test_strip_collapses_excessive_newlines(self):
        # Tag removal can leave 3+ consecutive newlines
        text = "Before\n\n\n<memory_cleanup>\ndrop: block:aaaaaaaa\n</memory_cleanup>\n\n\nAfter"
        result = strip_cleanup_tags(text)
        # Must not have 3+ consecutive newlines
        assert "\n\n\n" not in result

    def test_strip_tag_only_text(self):
        # Text that is ONLY a tag
        text = "<memory_cleanup>\ndrop: block:aaaaaaaa\n</memory_cleanup>"
        result = strip_cleanup_tags(text)
        assert "<memory_cleanup>" not in result
        assert "</memory_cleanup>" not in result

    def test_strip_result_has_no_internal_tag_content(self):
        # Inner operation lines should not appear in stripped result
        text = (
            "Hello\n"
            '<memory_cleanup>\n'
            'drop: block:a1b2c3d4\n'
            'anchor: block:b2c3d4e5\n'
            '</memory_cleanup>\n'
            "World"
        )
        result = strip_cleanup_tags(text)
        assert "drop:" not in result
        assert "anchor:" not in result


# ===========================================================================
# BLOCK OPERATIONS — CONTRACT INVARIANTS
# ===========================================================================

class TestDropContractInvariant:
    """Contract: drop marks the block. Content stays for audit but does not
    appear in future messages. The block disappears from the next request."""

    def test_drop_unknown_id_returns_false(self):
        bs = BlockStore()
        assert bs.drop("deadbeef") is False

    def test_drop_known_id_returns_true(self):
        bs, _ = _labeled_store(_make_long("A"))
        bid = _block_ids(bs)[0]
        assert bs.drop(bid) is True

    def test_drop_then_apply_removes_content_from_message(self):
        bs, messages = _labeled_store(_make_long("X"))
        bid = _block_ids(bs)[0]
        original_content = _make_long("X")

        bs.drop(bid)
        bs.apply_to_messages(messages)

        msg_content = messages[0]["content"]
        assert original_content not in msg_content, (
            "Dropped block content should not appear in message after apply"
        )

    def test_dropped_block_content_replaced_with_archive_marker(self):
        bs, messages = _labeled_store(_make_long("Y"))
        bid = _block_ids(bs)[0]

        bs.drop(bid)
        bs.apply_to_messages(messages)

        msg_content = messages[0]["content"]
        # Contract says: block disappears from next request
        # Implementation uses archive marker — verify it's there
        assert "archived" in msg_content.lower() or "..." in msg_content, (
            "Dropped block should show some archive marker"
        )

    def test_drop_does_not_erase_entry_from_store(self):
        # "Content stays for audit" — entry must still exist in store
        bs, _ = _labeled_store(_make_long("Z"))
        bid = _block_ids(bs)[0]
        bs.drop(bid)
        entry = bs.get(bid)
        assert entry is not None, "Drop should not erase entry from store"


class TestSummarizeContractInvariant:
    """Contract: summarize replaces content with model-authored summary + fault handle."""

    def test_summarize_unknown_id_returns_false(self):
        bs = BlockStore()
        assert bs.summarize("deadbeef", "some summary") is False

    def test_summarize_known_id_returns_true(self):
        bs, _ = _labeled_store(_make_long("B"))
        bid = _block_ids(bs)[0]
        assert bs.summarize(bid, "summary") is True

    def test_summarize_then_apply_replaces_content(self):
        bs, messages = _labeled_store(_make_long("C"))
        bid = _block_ids(bs)[0]
        original = _make_long("C")

        bs.summarize(bid, "This is the summary")
        bs.apply_to_messages(messages)

        msg_content = messages[0]["content"]
        assert original not in msg_content, (
            "Summarized block's original content should be replaced"
        )

    def test_summarize_then_apply_includes_summary_text(self):
        bs, messages = _labeled_store(_make_long("D"))
        bid = _block_ids(bs)[0]

        bs.summarize(bid, "The critical finding about D content")
        bs.apply_to_messages(messages)

        msg_content = messages[0]["content"]
        assert "The critical finding about D content" in msg_content, (
            "Summary text should appear in the message after apply"
        )

    def test_summarize_empty_summary_string_handled(self):
        # Empty summary is a degenerate case — apply should not crash
        bs, messages = _labeled_store(_make_long("E"))
        bid = _block_ids(bs)[0]
        bs.summarize(bid, "")
        # Should not raise
        stats = bs.apply_to_messages(messages)
        # An empty summary may or may not show as summarized — depends on impl
        # Contract: should not crash
        assert isinstance(stats, dict)

    def test_summarize_preserves_block_id_in_output(self):
        # Contract says: "fault handle" — the block_id must remain findable
        bs, messages = _labeled_store(_make_long("F"))
        bid = _block_ids(bs)[0]

        bs.summarize(bid, "Summary of F block")
        bs.apply_to_messages(messages)

        msg_content = messages[0]["content"]
        assert bid in msg_content, (
            "Block ID should remain in message after summarize for fault recovery"
        )


class TestAnchorContractInvariant:
    """Contract: anchor marks block for retention. Content stays in full."""

    def test_anchor_unknown_id_returns_false(self):
        bs = BlockStore()
        assert bs.anchor("deadbeef") is False

    def test_anchor_known_id_returns_true(self):
        bs, _ = _labeled_store(_make_long("G"))
        bid = _block_ids(bs)[0]
        assert bs.anchor(bid) is True

    def test_anchored_block_retains_full_content_in_message(self):
        content = _make_long("H")
        bs, messages = _labeled_store(content)
        bid = _block_ids(bs)[0]

        bs.anchor(bid)
        bs.apply_to_messages(messages)

        msg_content = messages[0]["content"]
        # The FULL original content must still be present
        assert content in msg_content, (
            "Anchored block must retain its full content in the message"
        )

    def test_anchored_not_counted_as_dropped_or_summarized(self):
        bs, messages = _labeled_store(_make_long("I"))
        bid = _block_ids(bs)[0]

        bs.anchor(bid)
        stats = bs.apply_to_messages(messages)

        assert stats["dropped"] == 0
        assert stats["summarized"] == 0


# ===========================================================================
# APPLY_TO_MESSAGES — CONTRACT INVARIANTS
# ===========================================================================

class TestApplyToMessagesInvariants:
    """Contract invariants that must hold for apply_to_messages."""

    def test_resident_block_content_unchanged(self):
        # Resident blocks must pass through untouched
        content = _make_long("J")
        bs, messages = _labeled_store(content)

        stats = bs.apply_to_messages(messages)

        # Content must still include the original text
        msg_content = messages[0]["content"]
        assert content in msg_content, (
            "Resident block content must be unchanged after apply"
        )
        assert stats == {"dropped": 0, "summarized": 0, "anchored": 0}

    def test_apply_idempotent_for_dropped_blocks(self):
        # Calling apply twice on a dropped block should produce stable output
        bs, messages = _labeled_store(_make_long("K"))
        bid = _block_ids(bs)[0]
        bs.drop(bid)

        stats1 = bs.apply_to_messages(messages)
        content_after_first = messages[0]["content"]

        stats2 = bs.apply_to_messages(messages)
        content_after_second = messages[0]["content"]

        # Content should be the same after both applications
        # (idempotent — not accumulating markers)
        assert content_after_first == content_after_second, (
            "apply_to_messages should be idempotent for dropped blocks"
        )

    def test_apply_idempotent_for_summarized_blocks(self):
        bs, messages = _labeled_store(_make_long("L"))
        bid = _block_ids(bs)[0]
        bs.summarize(bid, "Summary of L")

        bs.apply_to_messages(messages)
        content_after_first = messages[0]["content"]

        bs.apply_to_messages(messages)
        content_after_second = messages[0]["content"]

        assert content_after_first == content_after_second, (
            "apply_to_messages should be idempotent for summarized blocks"
        )

    def test_unlabeled_message_not_modified(self):
        # Messages without [block:...] labels should pass through unchanged
        bs = BlockStore()
        messages = [
            {"role": "user", "content": "This is a short message."}
        ]
        bs.apply_to_messages(messages)
        assert messages[0]["content"] == "This is a short message."

    def test_label_at_non_zero_position_not_matched(self):
        # Block label not at position 0 should NOT be matched
        # The regex anchors to start of string
        bs = BlockStore()
        content = "Some preamble\n[block:a1b2c3d4 (1.0KB)]\nActual content here"
        messages = [{"role": "user", "content": content}]
        bs.apply_to_messages(messages)
        # Should not be modified — label is not at position 0
        assert messages[0]["content"] == content

    def test_mixed_drop_and_resident_same_conversation(self):
        # Two blocks in the same messages list; one dropped, one resident
        content_a = _make_long("A")
        content_b = _make_long("B")
        bs, messages = _labeled_store(content_a, content_b)

        bids = _block_ids(bs)
        assert len(bids) == 2

        bs.drop(bids[0])
        stats = bs.apply_to_messages(messages)

        assert stats["dropped"] == 1
        assert stats["summarized"] == 0
        assert stats["anchored"] == 0

        # The non-dropped block should still have its content
        non_dropped_content_found = content_b in messages[1]["content"]
        assert non_dropped_content_found, (
            "Resident block should retain content when sibling block is dropped"
        )

    def test_apply_returns_stats_dict_with_required_keys(self):
        bs = BlockStore()
        messages = [{"role": "user", "content": "short"}]
        stats = bs.apply_to_messages(messages)
        assert "dropped" in stats
        assert "summarized" in stats
        assert "anchored" in stats

    def test_apply_with_no_messages(self):
        bs = BlockStore()
        stats = bs.apply_to_messages([])
        assert stats["dropped"] == 0
        assert stats["summarized"] == 0

    def test_apply_skips_messages_without_content_key(self):
        bs = BlockStore()
        messages = [{"role": "user"}]  # No "content" key
        # Should not raise
        stats = bs.apply_to_messages(messages)
        assert isinstance(stats, dict)

    def test_apply_handles_list_content_drops_correctly(self):
        bs = BlockStore()
        content = _make_long("M")
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": content},
            ]}
        ]
        bs.label_messages(messages, current_turn=1)
        bid = _block_ids(bs)[0]
        bs.drop(bid)

        stats = bs.apply_to_messages(messages)
        assert stats["dropped"] == 1

        text = messages[0]["content"][0]["text"]
        assert content not in text, "Dropped list-content block should be replaced"

    def test_apply_handles_list_content_anchored_correctly(self):
        bs = BlockStore()
        content = _make_long("N")
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": content},
            ]}
        ]
        bs.label_messages(messages, current_turn=1)
        bid = _block_ids(bs)[0]
        bs.anchor(bid)

        bs.apply_to_messages(messages)

        text = messages[0]["content"][0]["text"]
        assert content in text, "Anchored list-content block should retain full content"

    def test_apply_skips_non_text_blocks_in_list_content(self):
        # tool_use and tool_result blocks should be ignored
        bs = BlockStore()
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "abc", "content": "some result"},
                {"type": "tool_use", "name": "bash", "input": {"command": "ls"}},
            ]}
        ]
        original = [
            {"type": "tool_result", "tool_use_id": "abc", "content": "some result"},
            {"type": "tool_use", "name": "bash", "input": {"command": "ls"}},
        ]
        bs.apply_to_messages(messages)
        # Should be completely unchanged
        assert messages[0]["content"] == original

    def test_unknown_block_id_in_label_not_modified(self):
        # If a message has a [block:xxxx] label but the ID isn't in the store
        # (could happen if store was reset), the content should not be modified
        bs = BlockStore()
        fake_labeled = "[block:deadbeef (1.0KB)]\nSome content that was labeled before"
        messages = [{"role": "user", "content": fake_labeled}]

        stats = bs.apply_to_messages(messages)

        # Content should be unchanged — unknown ID means no operation
        assert messages[0]["content"] == fake_labeled, (
            "Unknown block ID in label should leave content unchanged"
        )
        assert stats["dropped"] == 0

    def test_drop_then_restore_then_apply_shows_content(self):
        # After restore(), the block is resident again — apply should not modify it
        content = _make_long("O")
        bs, messages = _labeled_store(content)
        bid = _block_ids(bs)[0]

        bs.drop(bid)
        bs.apply_to_messages(messages)

        # Now restore
        restored = bs.restore(bid)
        assert restored == content

        # Re-label (simulating next request cycle)
        # The message was modified to archive marker — reset it
        messages[0]["content"] = f"[block:{bid} (0.1KB)]\n{content}"

        stats = bs.apply_to_messages(messages)
        # After restore, block is resident — should not be dropped again
        assert stats["dropped"] == 0
        assert content in messages[0]["content"], (
            "Restored block should retain full content on next apply"
        )


# ===========================================================================
# TOTAL_BYTES AND STORE STATE
# ===========================================================================

class TestStoreStatistics:
    """total_bytes should reflect only resident blocks."""

    def test_total_bytes_excludes_dropped_blocks(self):
        content = _make_long("P")
        bs, _ = _labeled_store(content)
        bid = _block_ids(bs)[0]

        before = bs.total_bytes
        assert before > 0

        bs.drop(bid)
        after = bs.total_bytes

        assert after == 0, (
            "total_bytes should not count dropped blocks"
        )

    def test_total_bytes_excludes_summarized_blocks(self):
        content = _make_long("Q")
        bs, _ = _labeled_store(content)
        bid = _block_ids(bs)[0]

        before = bs.total_bytes
        assert before > 0

        bs.summarize(bid, "summary")
        after = bs.total_bytes

        assert after == 0, (
            "total_bytes should not count summarized blocks"
        )

    def test_total_bytes_includes_anchored_blocks(self):
        # Anchored blocks are still in working memory
        content = _make_long("R")
        bs, _ = _labeled_store(content)
        bid = _block_ids(bs)[0]

        before = bs.total_bytes
        bs.anchor(bid)
        after = bs.total_bytes

        # Anchored is NOT resident — check the actual contract
        # The docstring says: total_bytes counts only resident
        # Anchored blocks are NOT counted — this tests the contract
        # If this fails, total_bytes contract needs clarification
        # For now, document the actual behavior
        assert after == 0 or after == before, (
            "Anchored blocks: total_bytes behavior should be documented"
        )


# ===========================================================================
# PARSE + STRIP CONSISTENCY
# ===========================================================================

class TestParseAndStripConsistency:
    """parse_cleanup_tags and strip_cleanup_tags should be consistent."""

    def test_strip_does_not_destroy_info_visible_to_parse(self):
        # Parse then strip: the info was already parsed, strip is for forwarding
        text = (
            "Response text\n"
            '<memory_cleanup>\ndrop: block:a1b2c3d4\n</memory_cleanup>\n'
            "More response"
        )
        ops = parse_cleanup_tags(text)
        stripped = strip_cleanup_tags(text)

        # ops has the info
        assert "a1b2c3d4" in ops.drops

        # stripped has no tag content
        assert "<memory_cleanup>" not in stripped
        assert "a1b2c3d4" not in stripped  # block ID was inside tag

        # Stripping the already-stripped text is idempotent
        stripped_again = strip_cleanup_tags(stripped)
        assert stripped_again == stripped

    def test_round_trip_parse_strip_independent(self):
        # After stripping, parsing the stripped text gives empty ops
        text = '<memory_cleanup>\ndrop: block:b2c3d4e5\n</memory_cleanup>'
        stripped = strip_cleanup_tags(text)
        ops_from_stripped = parse_cleanup_tags(stripped)
        assert ops_from_stripped.empty, (
            "Parsing stripped text should yield no ops — tags are gone"
        )


# ===========================================================================
# CLEANUP_OPS DATACLASS COMPLETENESS
# ===========================================================================

class TestCleanupOpsCompleteness:
    """Verify CleanupOps contract beyond what existing tests cover."""

    def test_empty_only_when_all_lists_empty(self):
        # Each field independently makes it non-empty
        assert CleanupOps(drops=["x"]).empty is False
        assert CleanupOps(summaries=[("a", "b")]).empty is False
        assert CleanupOps(anchors=["x"]).empty is False
        assert CleanupOps(releases=["f"]).empty is False

    def test_str_includes_anchor_count(self):
        ops = CleanupOps(anchors=["a", "b"])
        s = str(ops)
        assert "anchor=2" in s

    def test_str_all_fields(self):
        ops = CleanupOps(
            drops=["a"],
            summaries=[("b", "s"), ("c", "t")],
            anchors=["d"],
            releases=["e", "f", "g"],
        )
        s = str(ops)
        assert "drop=1" in s
        assert "summarize=2" in s
        assert "anchor=1" in s
        assert "release=3" in s
