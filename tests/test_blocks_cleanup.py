"""Tests for BlockStore cleanup operations (Phase 2)."""

from pichay.blocks import BlockStore


def _make_store_with_blocks():
    """Create a BlockStore with some labeled messages."""
    bs = BlockStore()
    messages = [
        {"role": "user", "content": "A" * 300},
        {"role": "assistant", "content": "B" * 300},
        {"role": "user", "content": "C" * 300},
    ]
    bs.label_messages(messages, current_turn=1)
    return bs, messages


class TestDropOperation:

    def test_drop_marks_status(self):
        bs, messages = _make_store_with_blocks()
        block_id = list(bs._by_id.keys())[0]
        assert bs.drop(block_id)
        assert bs._by_id[block_id].status == "dropped"

    def test_drop_unknown_id(self):
        bs, _ = _make_store_with_blocks()
        assert not bs.drop("nonexist")

    def test_drop_preserves_original_content(self):
        bs, _ = _make_store_with_blocks()
        block_id = list(bs._by_id.keys())[0]
        bs.drop(block_id)
        assert bs._by_id[block_id].original_content is not None


class TestSummarizeOperation:

    def test_summarize_sets_status_and_summary(self):
        bs, _ = _make_store_with_blocks()
        block_id = list(bs._by_id.keys())[0]
        assert bs.summarize(block_id, "This was about X")
        entry = bs._by_id[block_id]
        assert entry.status == "summarized"
        assert entry.summary == "This was about X"

    def test_summarize_unknown_id(self):
        bs, _ = _make_store_with_blocks()
        assert not bs.summarize("nonexist", "summary")


class TestAnchorOperation:

    def test_anchor_marks_status(self):
        bs, _ = _make_store_with_blocks()
        block_id = list(bs._by_id.keys())[0]
        assert bs.anchor(block_id)
        assert bs._by_id[block_id].status == "anchored"

    def test_anchor_unknown_id(self):
        bs, _ = _make_store_with_blocks()
        assert not bs.anchor("nonexist")


class TestApplyToMessages:

    def test_apply_drops_content(self):
        bs, messages = _make_store_with_blocks()
        block_id = list(bs._by_id.keys())[0]
        bs.drop(block_id)

        stats = bs.apply_to_messages(messages)
        assert stats["dropped"] == 1

        # The dropped message should contain archived marker
        found_archived = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "...archived" in content:
                found_archived = True
        assert found_archived

    def test_apply_summarizes_content(self):
        bs, messages = _make_store_with_blocks()
        block_id = list(bs._by_id.keys())[0]
        bs.summarize(block_id, "This block discussed topic X")

        stats = bs.apply_to_messages(messages)
        assert stats["summarized"] == 1

        # The summarized message should contain the summary
        found_summary = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "topic X" in content:
                found_summary = True
        assert found_summary

    def test_apply_preserves_anchored(self):
        bs, messages = _make_store_with_blocks()
        block_id = list(bs._by_id.keys())[0]
        original_content = bs._by_id[block_id].original_content
        bs.anchor(block_id)

        stats = bs.apply_to_messages(messages)
        assert stats["anchored"] == 1

        # Content should still be there
        found = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and original_content[:50] in content:
                found = True
        assert found

    def test_apply_no_ops_on_resident(self):
        bs, messages = _make_store_with_blocks()
        stats = bs.apply_to_messages(messages)
        assert stats == {"dropped": 0, "summarized": 0, "anchored": 0}

    def test_apply_handles_list_content(self):
        bs = BlockStore()
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "D" * 300},
            ]},
        ]
        bs.label_messages(messages, current_turn=1)
        block_id = list(bs._by_id.keys())[0]
        bs.drop(block_id)

        stats = bs.apply_to_messages(messages)
        assert stats["dropped"] == 1

    def test_restore_after_drop(self):
        bs, _ = _make_store_with_blocks()
        block_id = list(bs._by_id.keys())[0]
        original = bs._by_id[block_id].original_content

        bs.drop(block_id)
        assert bs._by_id[block_id].status == "dropped"

        restored = bs.restore(block_id)
        assert restored == original
        assert bs._by_id[block_id].status == "resident"
