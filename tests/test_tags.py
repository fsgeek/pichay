"""Tests for cleanup tag parser."""

from pichay.tags import CleanupOps, parse_cleanup_tags, strip_cleanup_tags


class TestParseCleanupTags:
    """Test parsing of <memory_cleanup> tags."""

    def test_parse_drop(self):
        text = "<memory_cleanup>\ndrop: block:a3f2b901\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.drops == ["a3f2b901"]

    def test_parse_summarize(self):
        text = '<memory_cleanup>\nsummarize: block:7e9d4c12 "Summary text here"\n</memory_cleanup>'
        ops = parse_cleanup_tags(text)
        assert ops.summaries == [("7e9d4c12", "Summary text here")]

    def test_parse_anchor(self):
        text = "<memory_cleanup>\nanchor: block:c8ad36b2\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.anchors == ["c8ad36b2"]

    def test_parse_release(self):
        text = "<memory_cleanup>\nrelease: src/foo.py, src/bar.py\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.releases == ["src/foo.py", "src/bar.py"]

    def test_parse_multiple_operations(self):
        text = (
            "<memory_cleanup>\n"
            "drop: block:aaaaaaaa\n"
            "drop: block:bbbbbbbb\n"
            'summarize: block:cccccccc "A summary"\n'
            "anchor: block:dddddddd\n"
            "release: file1.py, file2.py\n"
            "</memory_cleanup>"
        )
        ops = parse_cleanup_tags(text)
        assert len(ops.drops) == 2
        assert len(ops.summaries) == 1
        assert len(ops.anchors) == 1
        assert len(ops.releases) == 2

    def test_parse_multiple_tags(self):
        text = (
            "Some text.\n"
            "<memory_cleanup>\ndrop: block:aaaaaaaa\n</memory_cleanup>\n"
            "More text.\n"
            "<memory_cleanup>\ndrop: block:bbbbbbbb\n</memory_cleanup>"
        )
        ops = parse_cleanup_tags(text)
        assert ops.drops == ["aaaaaaaa", "bbbbbbbb"]

    def test_parse_no_tags(self):
        text = "Just regular text with no tags."
        ops = parse_cleanup_tags(text)
        assert ops.empty

    def test_parse_empty_tag(self):
        text = "<memory_cleanup>\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.empty

    def test_parse_unknown_operation(self):
        text = "<memory_cleanup>\nfrobnicate: block:aaaaaaaa\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.empty

    def test_parse_12_char_block_id(self):
        text = "<memory_cleanup>\ndrop: block:a3f2b901cdef\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.drops == ["a3f2b901cdef"]

    def test_parse_release_single_path(self):
        text = "<memory_cleanup>\nrelease: single_file.py\n</memory_cleanup>"
        ops = parse_cleanup_tags(text)
        assert ops.releases == ["single_file.py"]

    def test_parse_inline_with_surrounding_text(self):
        text = (
            "Here is my analysis.\n\n"
            "<memory_cleanup>\ndrop: block:aaaaaaaa\n</memory_cleanup>\n\n"
            "And here is my conclusion."
        )
        ops = parse_cleanup_tags(text)
        assert ops.drops == ["aaaaaaaa"]


class TestStripCleanupTags:
    """Test stripping of <memory_cleanup> tags."""

    def test_strip_single_tag(self):
        text = "Before.\n\n<memory_cleanup>\ndrop: block:aaaaaaaa\n</memory_cleanup>\n\nAfter."
        result = strip_cleanup_tags(text)
        assert "<memory_cleanup>" not in result
        assert "Before." in result
        assert "After." in result

    def test_strip_preserves_content(self):
        text = "No tags here."
        result = strip_cleanup_tags(text)
        assert result == "No tags here."

    def test_strip_multiple_tags(self):
        text = (
            "<memory_cleanup>\ndrop: block:aaaa0000\n</memory_cleanup>\n"
            "middle\n"
            "<memory_cleanup>\ndrop: block:bbbb0000\n</memory_cleanup>"
        )
        result = strip_cleanup_tags(text)
        assert "<memory_cleanup>" not in result
        assert "middle" in result

    def test_strip_reduces_excessive_newlines(self):
        text = "Before.\n\n\n<memory_cleanup>\ndrop: block:aaaa0000\n</memory_cleanup>\n\n\nAfter."
        result = strip_cleanup_tags(text)
        # Should not have 4+ consecutive newlines
        assert "\n\n\n\n" not in result


class TestCleanupOps:
    """Test CleanupOps dataclass."""

    def test_empty(self):
        ops = CleanupOps()
        assert ops.empty

    def test_not_empty_with_drop(self):
        ops = CleanupOps(drops=["aaaa0000"])
        assert not ops.empty

    def test_str_representation(self):
        ops = CleanupOps(drops=["a"], summaries=[("b", "s")], releases=["f"])
        s = str(ops)
        assert "drop=1" in s
        assert "summarize=1" in s
        assert "release=1" in s

    def test_str_empty(self):
        assert str(CleanupOps()) == "no-ops"
