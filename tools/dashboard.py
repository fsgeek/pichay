#!/usr/bin/env python3
"""Live dashboard for pichay proxy sessions.

Reads proxy and page log JSONL files, plots working set curve,
fault rate, and compaction ratio over time.

Usage:
    python -m tools.dashboard logs/proxy_*.jsonl [logs/pages_*.jsonl]
    python -m tools.dashboard --live logs/proxy_CURRENT.jsonl logs/pages_CURRENT.jsonl
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def parse_proxy_log(path: str) -> list[dict]:
    """Extract per-request metrics from proxy JSONL.

    Working set data comes from 'compaction' records which have
    messages_bytes_before/after. Request records without compaction
    use messages.messages_total_bytes for the uncompacted size.
    """
    records = []
    call_index = 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = datetime.fromisoformat(rec["timestamp"])

            if rec["type"] == "compaction":
                before_bytes = rec["messages_bytes_before"]
                after_bytes = rec["messages_bytes_after"]
                records.append({
                    "timestamp": ts,
                    "before_bytes": before_bytes,
                    "after_bytes": after_bytes,
                    "before_kb": before_bytes / 1024,
                    "after_kb": after_bytes / 1024,
                    "est_tokens": int(after_bytes / 4.15),
                    "evicted": rec.get("evicted", 0),
                    "bytes_saved": rec.get("bytes_saved", 0),
                    "reduction_pct": rec.get("reduction_pct", 0),
                    "cumulative_faults": rec.get("cumulative_faults", 0),
                    "cumulative_evictions": rec.get("cumulative_evictions", 0),
                    "has_compaction": True,
                })
            elif rec["type"] == "request":
                msg_bytes = rec.get("messages", {}).get(
                    "messages_total_bytes", 0)
                records.append({
                    "timestamp": ts,
                    "before_bytes": msg_bytes,
                    "after_bytes": msg_bytes,
                    "before_kb": msg_bytes / 1024,
                    "after_kb": msg_bytes / 1024,
                    "est_tokens": int(msg_bytes / 4.15),
                    "evicted": 0,
                    "bytes_saved": 0,
                    "reduction_pct": 0,
                    "has_compaction": False,
                })
    return records


def parse_page_log(path: str) -> list[dict]:
    """Extract eviction and fault events from page JSONL."""
    events = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = datetime.fromisoformat(rec["timestamp"])
            events.append({
                "timestamp": ts,
                "type": rec.get("type", "unknown"),
                "tool_name": rec.get("tool_name", ""),
                "original_size": rec.get("original_size", 0),
                "summary_size": rec.get("summary_size", 0),
            })
    return events


def plot_dashboard(requests: list[dict], events: list[dict],
                   output: str | None = None):
    """Generate the dashboard figure."""
    if not requests:
        print("No request data found.", file=sys.stderr)
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Pichay Proxy Dashboard", fontsize=14, fontweight="bold")

    timestamps = [r["timestamp"] for r in requests]
    call_numbers = list(range(1, len(requests) + 1))

    # --- Panel 1: Working Set Curve ---
    ax1 = axes[0]
    before_kb = [r["before_kb"] for r in requests]
    after_kb = [r["after_kb"] for r in requests]
    tokens = [r["est_tokens"] for r in requests]

    ax1.fill_between(call_numbers, before_kb, after_kb,
                     alpha=0.3, color="red", label="Evicted")
    ax1.fill_between(call_numbers, after_kb, 0,
                     alpha=0.3, color="blue", label="Working set")
    ax1.plot(call_numbers, after_kb, color="blue", linewidth=1.5,
             label="Forwarded (KB)")
    ax1.plot(call_numbers, before_kb, color="red", linewidth=0.8,
             alpha=0.6, label="Pre-eviction (KB)")

    # Token cap line
    cap_kb = 200_000 * 4.15 / 1024  # rough inverse
    ax1.axhline(y=cap_kb, color="black", linestyle="--", alpha=0.3,
                label=f"200K token cap (~{cap_kb:.0f}KB)")

    ax1.set_ylabel("Message payload (KB)")
    ax1.set_title("Working Set Curve")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Secondary y-axis for tokens
    ax1_tok = ax1.twinx()
    ax1_tok.plot(call_numbers, tokens, color="green", linewidth=0,
                 alpha=0)  # invisible, just for axis
    ax1_tok.set_ylabel("Est. tokens", color="gray")
    ax1_tok.tick_params(axis="y", labelcolor="gray")
    if tokens:
        ax1_tok.set_ylim(0, max(max(before_kb) * 1024 / 4.15, max(tokens)) * 1.1)

    # --- Panel 2: Compaction Ratio ---
    ax2 = axes[1]
    reduction = [r["reduction_pct"] for r in requests]
    colors = ["green" if r > 0 else "gray" for r in reduction]
    ax2.bar(call_numbers, reduction, color=colors, alpha=0.6, width=1.0)
    ax2.set_ylabel("Reduction (%)")
    ax2.set_title("Per-Request Compaction Ratio")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Cumulative Faults ---
    ax3 = axes[2]
    if events:
        faults = [e for e in events if e["type"] == "fault"]
        evictions = [e for e in events if e["type"] == "eviction"]

        # Map events to call numbers by timestamp
        fault_times = [f["timestamp"] for f in faults]
        eviction_cumulative = []
        fault_cumulative = []
        eviction_count = 0
        fault_count = 0

        for i, req in enumerate(requests):
            t = req["timestamp"]
            while eviction_count < len(evictions) and evictions[eviction_count]["timestamp"] <= t:
                eviction_count += 1
            while fault_count < len(faults) and faults[fault_count]["timestamp"] <= t:
                fault_count += 1
            eviction_cumulative.append(eviction_count)
            fault_cumulative.append(fault_count)

        ax3.plot(call_numbers, eviction_cumulative, color="orange",
                 linewidth=1.5, label=f"Cumulative evictions ({eviction_count})")
        ax3.plot(call_numbers, fault_cumulative, color="red",
                 linewidth=2, label=f"Cumulative faults ({fault_count})")

        if eviction_count > 0:
            rate = fault_count / eviction_count * 100
            ax3.text(0.98, 0.95, f"Fault rate: {rate:.2f}%",
                     transform=ax3.transAxes, ha="right", va="top",
                     fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat"))

        ax3.legend(loc="upper left", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No page log data", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=12, color="gray")

    ax3.set_xlabel("API call number")
    ax3.set_ylabel("Count")
    ax3.set_title("Evictions and Faults")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}", file=sys.stderr)
    else:
        plt.show()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    proxy_path = args[0]
    page_path = args[1] if len(args) > 1 else None
    output = None

    # Check for --output flag
    for i, arg in enumerate(args):
        if arg == "--output" and i + 1 < len(args):
            output = args[i + 1]
            args = args[:i] + args[i+2:]
            break
        elif arg.startswith("--output="):
            output = arg.split("=", 1)[1]
            args = args[:i] + args[i+1:]
            break

    proxy_path = args[0]
    page_path = args[1] if len(args) > 1 else None

    requests = parse_proxy_log(proxy_path)
    events = parse_page_log(page_path) if page_path else []

    print(f"Parsed {len(requests)} requests, {len(events)} page events",
          file=sys.stderr)

    if not output:
        output = proxy_path.replace(".jsonl", "_dashboard.png")

    plot_dashboard(requests, events, output=output)


if __name__ == "__main__":
    main()
