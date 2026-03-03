"""Tests for pager eviction semantics.

Verifies the distinction between eviction (Read results, faultable)
and garbage collection (ephemeral tools, always safe).
"""

import time

from pichay.pager import PageEntry, PageStore, _eviction_key


class TestEvictionKey:
    """_eviction_key returns keys only for faultable tools."""

    def test_read_returns_file_path(self):
        assert _eviction_key("Read", {"file_path": "/foo.py"}) == "/foo.py"

    def test_read_no_path_returns_none(self):
        assert _eviction_key("Read", {}) is None

    def test_bash_returns_none(self):
        assert _eviction_key("Bash", {"command": "ls -la"}) is None

    def test_grep_returns_none(self):
        assert _eviction_key("Grep", {"pattern": "TODO", "path": "."}) is None

    def test_glob_returns_none(self):
        assert _eviction_key("Glob", {"pattern": "*.py"}) is None

    def test_web_fetch_returns_none(self):
        assert _eviction_key("WebFetch", {"url": "https://example.com"}) is None

    def test_web_search_returns_none(self):
        assert _eviction_key("WebSearch", {"query": "test"}) is None

    def test_agent_returns_none(self):
        assert _eviction_key("Agent", {"prompt": "do something"}) is None

    def test_unknown_tool_returns_none(self):
        assert _eviction_key("FutureTool", {"x": 1}) is None


def _make_entry(
    tool_use_id: str,
    tool_name: str = "Read",
    tool_input: dict | None = None,
    original_size: int = 1000,
    summary: str = "[evicted]",
    turn_index: int = 0,
    turns_from_end: int = 5,
) -> PageEntry:
    return PageEntry(
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        tool_input=tool_input or {"file_path": "/test.py"},
        original_content="x" * original_size,
        original_size=original_size,
        summary=summary,
        evicted_at=time.monotonic(),
        turn_index=turn_index,
        turns_from_end=turns_from_end,
    )


class TestPageStoreEvictionCounting:
    """Re-evictions don't inflate counters."""

    def test_first_eviction_counts(self):
        ps = PageStore()
        ps.store(_make_entry("id1"))
        assert ps.unique_evictions == 1

    def test_re_eviction_does_not_count(self):
        ps = PageStore()
        entry = _make_entry("id1")
        ps.store(entry)
        ps.store(entry)  # Same tool_use_id
        ps.store(entry)  # Again
        assert ps.unique_evictions == 1

    def test_different_ids_count_separately(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_input={"file_path": "/a.py"}))
        ps.store(_make_entry("id2", tool_input={"file_path": "/b.py"}))
        assert ps.unique_evictions == 2


class TestGarbageCollectionSplit:
    """Ephemeral tools tracked as GC, not evictions."""

    def test_bash_is_gc(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Bash", tool_input={"command": "ls"}))
        assert ps.gc_count == 1
        assert ps.unique_evictions == 0

    def test_grep_is_gc(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Grep", tool_input={"pattern": "x"}))
        assert ps.gc_count == 1
        assert ps.unique_evictions == 0

    def test_glob_is_gc(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Glob", tool_input={"pattern": "*.py"}))
        assert ps.gc_count == 1
        assert ps.unique_evictions == 0

    def test_read_is_eviction(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Read", tool_input={"file_path": "/x.py"}))
        assert ps.unique_evictions == 1
        assert ps.gc_count == 0

    def test_mixed_counting(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Read", tool_input={"file_path": "/x.py"}))
        ps.store(_make_entry("id2", tool_name="Bash", tool_input={"command": "ls"}))
        ps.store(_make_entry("id3", tool_name="Bash", tool_input={"command": "pwd"}))
        ps.store(_make_entry("id4", tool_name="Read", tool_input={"file_path": "/y.py"}))
        assert ps.unique_evictions == 2
        assert ps.gc_count == 2

    def test_bytes_tracked_separately(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Read", tool_input={"file_path": "/x.py"}, original_size=500))
        ps.store(_make_entry("id2", tool_name="Bash", tool_input={"command": "ls"}, original_size=300))
        assert ps.eviction_bytes_saved > 0
        assert ps.gc_bytes_saved > 0
        assert ps.total_bytes_saved == ps.eviction_bytes_saved + ps.gc_bytes_saved


class TestFaultDetection:
    """Faults only for Read, not ephemeral tools."""

    def test_read_fault_detected(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Read", tool_input={"file_path": "/foo.py"}))

        # Simulate assistant requesting the same file with a new tool_use_id
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "new_id",
                        "name": "Read",
                        "input": {"file_path": "/foo.py"},
                    }
                ],
            }
        ]
        faults = ps.detect_faults(messages)
        assert len(faults) == 1
        assert faults[0].tool_name == "Read"

    def test_bash_rerun_not_a_fault(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Bash", tool_input={"command": "ls"}))

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "new_id",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            }
        ]
        faults = ps.detect_faults(messages)
        assert len(faults) == 0

    def test_fault_rate_uses_unique_evictions(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Read", tool_input={"file_path": "/a.py"}))
        ps.store(_make_entry("id2", tool_name="Read", tool_input={"file_path": "/b.py"}))
        ps.store(_make_entry("id3", tool_name="Bash", tool_input={"command": "ls"}))

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "new_id",
                        "name": "Read",
                        "input": {"file_path": "/a.py"},
                    }
                ],
            }
        ]
        ps.detect_faults(messages)
        # 1 fault / 2 unique evictions = 0.5
        assert ps.fault_rate == 0.5
        # GC doesn't affect fault rate
        assert ps.gc_count == 1


class TestPageStoreSummary:
    """Summary exposes correct metrics."""

    def test_summary_has_split_counters(self):
        ps = PageStore()
        ps.store(_make_entry("id1", tool_name="Read", tool_input={"file_path": "/x.py"}))
        ps.store(_make_entry("id2", tool_name="Bash", tool_input={"command": "ls"}))
        s = ps.summary()
        assert "unique_evictions" in s
        assert "gc_count" in s
        assert s["unique_evictions"] == 1
        assert s["gc_count"] == 1
