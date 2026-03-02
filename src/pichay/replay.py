"""Offline replay of context paging on proxy logs.

Reads proxy JSONL logs (from observe-mode runs), reconstructs the
messages array at each API call turn, applies pager compaction, and
reports what WOULD have happened: how much content would be evicted,
and whether the model's actual next actions would have triggered page
faults.

This enables "what-if" analysis: run once without compaction, then
replay offline with different thresholds to find optimal parameters
without burning API credits.

Usage:
    # Replay one session
    python -m pichay.replay logs/proxy_20260302_024551.jsonl

    # Replay with custom thresholds
    python -m pichay.replay --age-threshold 6 --min-size 1000 logs/*.jsonl

    # JSON output for downstream analysis
    python -m pichay.replay --json logs/proxy_*.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from pichay.eval import parse_proxy_log
from pichay.pager import PageStore, compact_messages


@dataclass
class ReplayTurn:
    """Results of simulated compaction at one API call."""

    turn: int
    timestamp: str
    message_count: int
    bytes_original: int
    bytes_compacted: int
    bytes_saved: int
    reduction_pct: float
    evictions: int
    faults: int
    cumulative_evictions: int
    cumulative_faults: int


@dataclass
class SessionReplay:
    """Aggregate replay results for one proxy log."""

    log_path: str
    total_turns: int = 0
    total_evictions: int = 0
    total_bytes_saved: int = 0
    total_bytes_original: int = 0
    total_faults: int = 0
    turns: list[ReplayTurn] = field(default_factory=list)

    @property
    def fault_rate(self) -> float:
        if self.total_evictions == 0:
            return 0.0
        return self.total_faults / self.total_evictions

    @property
    def reduction_pct(self) -> float:
        if self.total_bytes_original == 0:
            return 0.0
        return (self.total_bytes_saved / self.total_bytes_original) * 100


def replay_session(
    path: Path,
    age_threshold: int = 4,
    min_size: int = 500,
) -> SessionReplay:
    """Replay compaction on a proxy log session.

    For each API call in the log, reconstructs the messages array,
    applies cumulative compaction (simulating what would have happened
    if the pager had been active from the start), and checks for
    page faults against the model's actual next actions.
    """
    records = parse_proxy_log(path)
    result = SessionReplay(log_path=str(path))

    requests = [
        r for r in records
        if r.get("type") == "request" and "messages_full" in r
    ]

    if not requests:
        return result

    page_store = PageStore()

    for turn_idx, req in enumerate(requests):
        messages = copy.deepcopy(req["messages_full"])
        bytes_original = len(json.dumps(messages).encode("utf-8"))

        # Apply previous evictions (simulate cumulative compaction)
        _apply_evictions(messages, page_store)

        # Detect faults: did the model re-request evicted content?
        faults = page_store.detect_faults(messages)

        # Run compaction on this turn
        stats = compact_messages(
            messages,
            age_threshold=age_threshold,
            min_size=min_size,
            page_store=page_store,
        )

        bytes_compacted = len(json.dumps(messages).encode("utf-8"))
        bytes_saved = bytes_original - bytes_compacted

        turn = ReplayTurn(
            turn=turn_idx + 1,
            timestamp=req.get("timestamp", ""),
            message_count=len(req["messages_full"]),
            bytes_original=bytes_original,
            bytes_compacted=bytes_compacted,
            bytes_saved=bytes_saved,
            reduction_pct=(
                bytes_saved / bytes_original * 100
                if bytes_original > 0
                else 0.0
            ),
            evictions=stats.evicted_count,
            faults=len(faults),
            cumulative_evictions=page_store.cumulative_evictions,
            cumulative_faults=len(page_store.faults),
        )
        result.turns.append(turn)
        result.total_turns += 1
        result.total_evictions += stats.evicted_count
        result.total_bytes_saved += bytes_saved
        result.total_bytes_original += bytes_original
        result.total_faults += len(faults)

    return result


def _apply_evictions(messages: list[dict], page_store: PageStore) -> None:
    """Replace tool result content for previously evicted results."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for i, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id", "")
            entry = page_store.retrieve(tool_use_id)
            if entry is not None:
                content[i] = {**block, "content": entry.summary}


def print_session_report(s: SessionReplay) -> None:
    """Print a human-readable replay report."""
    print(f"\n{'='*60}")
    print(f"Replay: {s.log_path}")
    print(f"{'='*60}")

    print(f"\nTurns:               {s.total_turns:>10,}")
    print(f"Evictions:           {s.total_evictions:>10,}")
    print(f"Bytes saved:         {s.total_bytes_saved:>10,}")
    print(f"Bytes original:      {s.total_bytes_original:>10,}")
    print(f"Reduction:           {s.reduction_pct:>9.1f}%")
    print(f"Faults:              {s.total_faults:>10,}")
    print(f"Fault rate:          {s.fault_rate:>9.2%}")

    if not s.turns:
        return

    # Per-turn detail
    print(f"\nPer-turn:")
    print(
        f"  {'Turn':>4s}  {'Original':>10s}  {'Compacted':>10s}  "
        f"{'Saved':>10s}  {'Reduc%':>6s}  {'Evict':>5s}  {'Fault':>5s}"
    )
    for t in s.turns:
        print(
            f"  T{t.turn:>3d}  {t.bytes_original:>10,}  "
            f"{t.bytes_compacted:>10,}  {t.bytes_saved:>10,}  "
            f"{t.reduction_pct:>5.1f}%  "
            f"{t.evictions:>5d}  {t.faults:>5d}"
        )

    # Cumulative curve (final state)
    last = s.turns[-1]
    print(f"\nFinal turn context: {last.bytes_original:,} → "
          f"{last.bytes_compacted:,} bytes "
          f"({last.reduction_pct:.1f}% reduction)")


def print_aggregate(sessions: list[SessionReplay]) -> None:
    """Print aggregate stats across multiple sessions."""
    if not sessions:
        return

    total_turns = sum(s.total_turns for s in sessions)
    total_evictions = sum(s.total_evictions for s in sessions)
    total_bytes_saved = sum(s.total_bytes_saved for s in sessions)
    total_bytes_original = sum(s.total_bytes_original for s in sessions)
    total_faults = sum(s.total_faults for s in sessions)

    fault_rate = (
        total_faults / total_evictions if total_evictions > 0 else 0.0
    )
    reduction_pct = (
        total_bytes_saved / total_bytes_original * 100
        if total_bytes_original > 0
        else 0.0
    )

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({len(sessions)} sessions)")
    print(f"{'='*60}")
    print(f"Total turns:         {total_turns:>10,}")
    print(f"Total evictions:     {total_evictions:>10,}")
    print(f"Total bytes saved:   {total_bytes_saved:>10,}")
    print(f"Total bytes original:{total_bytes_original:>10,}")
    print(f"Reduction:           {reduction_pct:>9.1f}%")
    print(f"Faults:              {total_faults:>10,}")
    print(f"Fault rate:          {fault_rate:>9.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline replay of context paging on proxy logs"
    )
    parser.add_argument(
        "logs",
        type=Path,
        nargs="+",
        help="Proxy JSONL log file(s) to replay",
    )
    parser.add_argument(
        "--age-threshold",
        type=int,
        default=4,
        help="Evict tool results older than N user-turns (default: 4)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=500,
        help="Don't evict results smaller than N bytes (default: 500)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    sessions: list[SessionReplay] = []
    for log_path in args.logs:
        if not log_path.exists():
            print(
                f"Warning: {log_path} not found, skipping.",
                file=sys.stderr,
            )
            continue
        sessions.append(replay_session(
            log_path,
            age_threshold=args.age_threshold,
            min_size=args.min_size,
        ))

    if args.json:
        for s in sessions:
            out = asdict(s)
            out["fault_rate"] = s.fault_rate
            out["reduction_pct"] = s.reduction_pct
            out["turn_count"] = len(out.pop("turns"))
            print(json.dumps(out))
        if len(sessions) > 1:
            total_e = sum(s.total_evictions for s in sessions)
            total_o = sum(s.total_bytes_original for s in sessions)
            agg = {
                "type": "aggregate",
                "sessions": len(sessions),
                "total_turns": sum(s.total_turns for s in sessions),
                "total_evictions": total_e,
                "total_bytes_saved": sum(
                    s.total_bytes_saved for s in sessions
                ),
                "total_bytes_original": total_o,
                "total_faults": sum(s.total_faults for s in sessions),
                "fault_rate": (
                    sum(s.total_faults for s in sessions) / total_e
                    if total_e > 0
                    else 0.0
                ),
                "reduction_pct": (
                    sum(s.total_bytes_saved for s in sessions) / total_o * 100
                    if total_o > 0
                    else 0.0
                ),
            }
            print(json.dumps(agg))
        return

    for s in sessions:
        print_session_report(s)

    if len(sessions) > 1:
        print_aggregate(sessions)


if __name__ == "__main__":
    main()
