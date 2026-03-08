"""Analyze Claude Code's cache behavior from Pichay's proxy logs.

Answers questions:
1. How does Claude Code set cache_control breakpoints?
2. How often does Claude Code compact/evict?
3. What's the cache hit rate through Pichay vs theoretical without mutation?
4. How does the message stream change between consecutive requests?

Reads from the proxy JSONL logs (messages_full captured per request).

Usage:
    python tools/analyze_claude_code_cache.py [log_dir]

Default log_dir: logs/
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def find_cache_controls(obj, path=""):
    """Recursively find all cache_control markers in a request."""
    results = []
    if isinstance(obj, dict):
        if "cache_control" in obj:
            results.append({
                "path": path,
                "cache_control": obj["cache_control"],
                "type": obj.get("type", "unknown"),
                "text_preview": str(obj.get("text", ""))[:80],
            })
        for k, v in obj.items():
            results.extend(find_cache_controls(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            results.extend(find_cache_controls(item, f"{path}[{i}]"))
    return results


def analyze_session(log_file):
    """Analyze one proxy session log."""
    requests = []
    responses = []

    for line in open(log_file):
        try:
            r = json.loads(line)
            if r.get("type") == "request":
                requests.append(r)
            elif r.get("type") in ("response", "response_stream"):
                responses.append(r)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if not requests:
        return None

    session_id = requests[0].get("session", "unknown")

    # --- Cache control analysis ---
    breakpoint_locations = Counter()
    breakpoint_types = Counter()
    for req in requests:
        system = req.get("system_prompt_full", req.get("system", ""))
        messages = req.get("messages_full", req.get("messages", []))

        ccs = find_cache_controls({"system": system, "messages": messages})
        for cc in ccs:
            breakpoint_locations[cc["path"]] += 1
            bp_type = cc["cache_control"]
            breakpoint_types[json.dumps(bp_type, sort_keys=True)] += 1

    # --- Message stream changes ---
    msg_count_changes = []
    content_changes = []
    prev_msgs = None
    prev_msg_count = 0
    compaction_events = 0
    eviction_events = 0

    for req in requests:
        msgs = req.get("messages_full", [])
        msg_count = len(msgs)
        msg_count_changes.append(msg_count)

        if prev_msgs is not None:
            if msg_count < prev_msg_count:
                # Message count dropped — compaction or eviction
                compaction_events += 1
            elif msg_count == prev_msg_count and prev_msgs:
                # Same count but content might differ
                for i in range(min(len(msgs), len(prev_msgs))):
                    if json.dumps(msgs[i]) != json.dumps(prev_msgs[i]):
                        content_changes.append({
                            "request_idx": len(msg_count_changes) - 1,
                            "message_idx": i,
                            "distance_from_end": msg_count - i,
                        })
                        eviction_events += 1
                        break  # count once per request

        prev_msgs = msgs
        prev_msg_count = msg_count

    # --- Cache hit rates ---
    total_read = 0
    total_create = 0
    total_input = 0
    for resp in responses:
        u = resp.get("usage", {})
        total_read += u.get("cache_read_input_tokens", 0)
        total_create += u.get("cache_creation_input_tokens", 0)
        total_input += u.get("input_tokens", 0)

    effective = total_read + total_create + total_input
    read_pct = (total_read / effective * 100) if effective else 0

    return {
        "session": session_id,
        "log_file": str(log_file.name),
        "requests": len(requests),
        "responses": len(responses),
        "breakpoint_locations": dict(breakpoint_locations.most_common(10)),
        "breakpoint_types": dict(breakpoint_types),
        "msg_count_range": (min(msg_count_changes), max(msg_count_changes)) if msg_count_changes else (0, 0),
        "compaction_events": compaction_events,
        "eviction_events": eviction_events,
        "content_change_distances": [c["distance_from_end"] for c in content_changes[:20]],
        "cache_read_tokens": total_read,
        "cache_create_tokens": total_create,
        "uncached_tokens": total_input,
        "effective_tokens": effective,
        "cache_read_pct": round(read_pct, 1),
    }


def main():
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")

    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    log_files = sorted(log_dir.glob("proxy_*.jsonl"))
    if not log_files:
        print(f"No proxy logs found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(log_files)} session logs in {log_dir}\n")

    all_results = []
    total_requests = 0
    total_compactions = 0
    total_evictions = 0
    total_read = 0
    total_create = 0
    total_input = 0
    all_bp_locations = Counter()
    all_bp_types = Counter()

    for f in log_files:
        result = analyze_session(f)
        if result is None:
            continue

        all_results.append(result)
        total_requests += result["requests"]
        total_compactions += result["compaction_events"]
        total_evictions += result["eviction_events"]
        total_read += result["cache_read_tokens"]
        total_create += result["cache_create_tokens"]
        total_input += result["uncached_tokens"]

        for loc, count in result["breakpoint_locations"].items():
            all_bp_locations[loc] += count
        for bt, count in result["breakpoint_types"].items():
            all_bp_types[bt] += count

    # --- Global summary ---
    effective = total_read + total_create + total_input
    read_pct = (total_read / effective * 100) if effective else 0

    print("=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)
    print(f"Sessions: {len(all_results)}")
    print(f"Total requests: {total_requests}")
    print(f"Compaction events (msg count dropped): {total_compactions}")
    print(f"Content mutation events (same count, changed content): {total_evictions}")
    print()

    print("Cache token totals:")
    print(f"  Read (hits):    {total_read:>14,} ({read_pct:.1f}%)")
    print(f"  Create (misses):{total_create:>14,}")
    print(f"  Uncached:       {total_input:>14,}")
    print(f"  Effective:      {effective:>14,}")
    print()

    print("Cache breakpoint locations (top 10):")
    for loc, count in all_bp_locations.most_common(10):
        print(f"  {loc}: {count}")
    print()

    print("Cache breakpoint types:")
    for bt, count in all_bp_types.most_common():
        print(f"  {bt}: {count}")
    print()

    # --- Per-session details for recent sessions ---
    print("=" * 80)
    print("PER-SESSION DETAILS (last 10)")
    print("=" * 80)
    for result in all_results[-10:]:
        print(f"\n  {result['log_file']}:")
        print(f"    Requests: {result['requests']}, "
              f"Msg range: {result['msg_count_range'][0]}-{result['msg_count_range'][1]}")
        print(f"    Compactions: {result['compaction_events']}, "
              f"Content mutations: {result['eviction_events']}")
        print(f"    Cache: {result['cache_read_pct']:.1f}% read "
              f"({result['cache_read_tokens']:,} read / "
              f"{result['cache_create_tokens']:,} create / "
              f"{result['uncached_tokens']:,} uncached)")
        if result["content_change_distances"]:
            print(f"    Mutation distances from end: {result['content_change_distances']}")
        if result["breakpoint_locations"]:
            top_bp = list(result["breakpoint_locations"].items())[:3]
            print(f"    Top breakpoints: {top_bp}")


if __name__ == "__main__":
    main()
