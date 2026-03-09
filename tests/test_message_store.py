"""Tests for pichay.message_store.MessageStore."""

from __future__ import annotations

import copy
import json

import pytest

from pichay.message_store import MessageStore, _fingerprint
from pichay.pager import CompactionStats


def _make_msg(role: str, content, **extra) -> dict:
    msg = {"role": role, "content": content}
    msg.update(extra)
    return msg


@pytest.fixture(autouse=True)
def stub_compaction(monkeypatch):
    """Stub compact_messages so MessageStore tests stay unit-scoped."""

    calls = []

    def _fake_compact(messages, age_threshold, min_size, page_store):
        calls.append(
            {
                "snapshot": copy.deepcopy(messages),
                "age_threshold": age_threshold,
                "min_size": min_size,
                "page_store": page_store,
            }
        )
        return CompactionStats()

    monkeypatch.setattr("pichay.message_store.compact_messages", _fake_compact)
    return calls


@pytest.fixture
def make_store(tmp_path):
    def _factory(session_id: str = "session"):
        log_path = tmp_path / f"{session_id}_violations.log"
        store = MessageStore(session_id, page_store=object(), log_path=log_path)
        return store, log_path

    return _factory


def test_basic_ingest_appends_physical_store(make_store):
    store, _ = make_store()
    incoming = [
        _make_msg("user", "hello"),
        _make_msg("assistant", "world"),
    ]

    result = store.ingest(copy.deepcopy(incoming))

    assert result.new_count == 2
    assert store.message_count == 2
    assert store.total_ingested == 2
    assert store.messages[0] == incoming[0]
    assert store.messages[0] is not incoming[0]
    assert store._client_to_physical == [0, 1]
    assert store._client_fps == [_fingerprint(msg) for msg in incoming]


def test_mutation_detection_logs_and_preserves_physical_store(make_store):
    store, log_path = make_store("mut")
    initial = [
        _make_msg("user", "hello"),
        _make_msg("assistant", "original reply"),
    ]
    store.ingest(copy.deepcopy(initial))

    mutated = copy.deepcopy(initial)
    mutated[1]["content"] = "mutated reply"

    result = store.ingest(mutated)

    assert result.mutations_detected == 1
    assert store.total_mutations == 1
    assert store.messages[1]["content"] == "original reply"

    payload = json.loads(log_path.read_text().strip())
    assert payload["kind"] == "mutation"
    assert payload["old_size"] == len("original reply")
    assert payload["new_size"] == len("mutated reply")


def test_deletion_absorption_preserves_physical_store_and_logs(make_store):
    store, log_path = make_store("del")
    initial = [
        _make_msg("user", "one"),
        _make_msg("assistant", "two"),
        _make_msg("user", "three"),
    ]
    store.ingest(copy.deepcopy(initial))

    truncated = copy.deepcopy(initial[:2])
    result = store.ingest(truncated)

    assert result.deletions_detected == 1
    assert store.total_deletions == 1
    assert store.total_client_deletions_absorbed == 1
    assert store.message_count == 3
    assert store._client_to_physical == [0, 1]

    payload = json.loads(log_path.read_text().strip())
    assert payload["kind"] == "deletion"
    assert payload["deleted_count"] == 1
    assert payload["deleted_messages"][0]["role"] == "user"
    assert payload["deleted_messages"][0]["preview"].startswith("three")


def test_client_mapping_survives_deletions_and_new_messages(make_store):
    store, _ = make_store("map")
    initial = [
        _make_msg("user", "first"),
        _make_msg("assistant", "second"),
        _make_msg("assistant", "third"),
    ]
    store.ingest(copy.deepcopy(initial))

    store.ingest(copy.deepcopy(initial[:2]))
    assert store._client_to_physical == [0, 1]

    after_delete = initial[:2] + [_make_msg("user", "post-delete new")]
    store.ingest(copy.deepcopy(after_delete))

    assert store._client_to_physical == [0, 1, 3]
    assert store.messages[3]["content"] == "post-delete new"


def test_cache_control_is_stripped_from_messages(make_store):
    store, _ = make_store("cache")
    incoming = [
        _make_msg(
            "assistant",
            [
                {
                    "type": "text",
                    "text": "cached block",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            cache_control={"type": "ephemeral"},
        )
    ]

    store.ingest(copy.deepcopy(incoming))

    stored = store.messages[0]
    assert "cache_control" not in stored
    assert "cache_control" not in stored["content"][0]


def test_fingerprint_stability_prevents_false_mutations(make_store):
    store, _ = make_store("fp")
    msgs = [
        _make_msg("user", "ask"),
        _make_msg("assistant", "answer"),
    ]
    store.ingest(copy.deepcopy(msgs))

    repeat = copy.deepcopy(msgs)
    result = store.ingest(repeat)

    assert result.mutations_detected == 0
    assert store.total_mutations == 0
    assert _fingerprint(msgs[0]) == _fingerprint(repeat[0])


def test_stats_tracking_across_operations(make_store):
    store, _ = make_store("stats")
    base = [
        _make_msg("user", "turn1"),
        _make_msg("assistant", "reply1"),
        _make_msg("user", "turn2"),
    ]
    store.ingest(copy.deepcopy(base))

    mutated = copy.deepcopy(base)
    mutated[1]["content"] = "reply1 updated"
    store.ingest(mutated)

    shortened = copy.deepcopy(mutated[:2])
    store.ingest(shortened)

    extended = shortened + [_make_msg("assistant", "after deletion new")]
    store.ingest(copy.deepcopy(extended))

    assert store.total_ingested == 4
    assert store.total_mutations == 1
    assert store.total_deletions == 1
    assert store.total_client_deletions_absorbed == 1


def test_multiple_turns_mixed_mutations_and_deletions(make_store):
    store, log_path = make_store("mix")
    turn1 = [
        _make_msg("user", "hi"),
        _make_msg("assistant", "there"),
    ]
    store.ingest(copy.deepcopy(turn1))

    turn2 = copy.deepcopy(turn1)
    turn2[0]["content"] = "hi mutated"
    turn2.append(_make_msg("user", "new turn"))
    res2 = store.ingest(turn2)

    turn3 = copy.deepcopy(turn2[:1])
    res3 = store.ingest(turn3)

    turn4 = turn3 + [
        _make_msg("assistant", "follow up"),
        _make_msg("user", "final"),
    ]
    res4 = store.ingest(copy.deepcopy(turn4))

    assert res2.mutations_detected == 1 and res2.new_count == 1
    assert res3.deletions_detected == 2
    assert res4.new_count == 2
    assert store.message_count == 5
    assert store._client_to_physical == [0, 3, 4]
    assert store.messages[3]["content"] == "follow up"

    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    kinds = [entry["kind"] for entry in entries]
    assert kinds.count("mutation") == 1
    assert kinds.count("deletion") == 1
