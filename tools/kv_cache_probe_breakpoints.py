"""Probe cache_control breakpoint behavior.

Tests:
1. How does Claude Code set its breakpoints? (Analyze from logged requests)
2. Can Pichay add breakpoints without causing a cache miss?
3. Does automatic caching (top-level cache_control) work alongside explicit?
4. Do breakpoints on messages advance the cached prefix?

Usage:
    python tools/kv_cache_probe_breakpoints.py

Requires ANTHROPIC_API_KEY in environment.
"""

import os
import sys
import time
import json
import httpx

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ANTHROPIC_API_KEY not set", file=sys.stderr)
    sys.exit(1)

URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

HEADERS = {
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}

SYSTEM_TEXT = (
    "You are a helpful assistant specializing in distributed systems. "
    "You provide concise, accurate answers about cache coherence, "
    "virtual memory, demand paging, and memory hierarchy optimization. "
) * 20

# 10 turn pairs = 20 messages
def make_turns(n=10):
    msgs = []
    for i in range(n):
        msgs.append({
            "role": "user",
            "content": f"TURN_{i:03d}_USER: Explain distributed concept {i}. "
                       f"Focus on the trade-offs between consistency and "
                       f"performance at this level of the hierarchy."
        })
        msgs.append({
            "role": "assistant",
            "content": f"TURN_{i:03d}_ASSISTANT: At level {i}, the key insight "
                       f"is that consistency guarantees must be weakened as "
                       f"we move further from the source of truth. Level {i} "
                       f"typically uses eventual consistency with conflict "
                       f"resolution based on vector clocks or similar."
        })
    return msgs


def send(body, label):
    """Send request, return usage stats."""
    resp = httpx.post(URL, headers=HEADERS, json=body, timeout=120)
    if resp.status_code != 200:
        print(f"  ERROR [{label}]: {resp.status_code} {resp.text[:300]}", file=sys.stderr)
        return None
    usage = resp.json().get("usage", {})
    r = {
        "label": label,
        "input": usage.get("input_tokens", 0),
        "create": usage.get("cache_creation_input_tokens", 0),
        "read": usage.get("cache_read_input_tokens", 0),
    }
    r["effective"] = r["input"] + r["create"] + r["read"]
    r["read_pct"] = round(r["read"] / r["effective"] * 100, 1) if r["effective"] else 0
    return r


def probe(label, body, results, delay=2):
    r = send(body, label)
    if r:
        results.append(r)
        hit = "HIT" if r["read"] > 0 else "MISS"
        print(f"  {r['label']:<55} {hit:>4}  "
              f"read={r['read']:>6,}  create={r['create']:>6,}  "
              f"input={r['input']:>5,}  ({r['read_pct']:>5.1f}%)")
    else:
        print(f"  {label:<55} FAILED")
    time.sleep(delay)


def run():
    results = []
    turns = make_turns(10)

    system_cached = [{"type": "text", "text": SYSTEM_TEXT, "cache_control": CACHE_CONTROL}]
    system_plain = [{"type": "text", "text": SYSTEM_TEXT}]

    print("=" * 80)
    print("EXPERIMENT A: Adding a breakpoint where there wasn't one")
    print("  Does adding cache_control to a message cause a miss?")
    print("=" * 80)

    # A1: System cached, messages plain, final query
    msgs_plain = turns + [{"role": "user", "content": "Summarize."}]
    body_a1 = {"model": MODEL, "max_tokens": 10, "system": system_cached, "messages": msgs_plain}
    probe("A1. system cached, msgs plain (cold)", body_a1, results)
    probe("A2. identical (warm)", body_a1, results)

    # A3: Same content, but add cache_control to message 10 (turn 5 assistant)
    msgs_with_bp = list(turns)
    msgs_with_bp[11] = {  # turn 5 assistant (index 11)
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": msgs_with_bp[11]["content"],
            "cache_control": CACHE_CONTROL,
        }]
    }
    msgs_with_bp.append({"role": "user", "content": "Summarize."})
    body_a3 = {"model": MODEL, "max_tokens": 10, "system": system_cached, "messages": msgs_with_bp}
    probe("A3. added breakpoint on msg 11 (new structure)", body_a3, results)
    probe("A4. same with breakpoint (warm)", body_a3, results)

    # A5: Remove the breakpoint again (back to plain)
    probe("A5. breakpoint removed (back to A1 structure)", body_a1, results)

    print()
    print("=" * 80)
    print("EXPERIMENT B: Automatic caching (top-level cache_control)")
    print("  Does top-level cache_control work? How does it interact?")
    print("=" * 80)

    # B1: Top-level automatic caching, no explicit breakpoints on messages
    body_b1 = {
        "model": MODEL, "max_tokens": 10,
        "cache_control": CACHE_CONTROL,
        "system": system_plain,  # no cache_control on system
        "messages": msgs_plain,
    }
    probe("B1. auto cache, no explicit breakpoints (cold)", body_b1, results)
    probe("B2. auto cache repeated (warm)", body_b1, results)

    # B3: Auto cache + system breakpoint (uses 2 of 4 slots)
    body_b3 = {
        "model": MODEL, "max_tokens": 10,
        "cache_control": CACHE_CONTROL,
        "system": system_cached,
        "messages": msgs_plain,
    }
    probe("B3. auto cache + system breakpoint (cold)", body_b3, results)
    probe("B4. auto + system repeated (warm)", body_b3, results)

    # B5: Auto cache, mutate last message
    msgs_mutated_tail = turns + [{"role": "user", "content": "Summarize please."}]
    body_b5 = {
        "model": MODEL, "max_tokens": 10,
        "cache_control": CACHE_CONTROL,
        "system": system_cached,
        "messages": msgs_mutated_tail,
    }
    probe("B5. auto + system, last msg mutated", body_b5, results)

    print()
    print("=" * 80)
    print("EXPERIMENT C: Growing conversation (simulates multi-turn)")
    print("  Does cache advance as conversation grows?")
    print("=" * 80)

    # Simulate conversation growth with auto-caching
    for n_turns in [3, 5, 7, 10]:
        msgs = make_turns(n_turns) + [{"role": "user", "content": "Summarize."}]
        body = {
            "model": MODEL, "max_tokens": 10,
            "cache_control": CACHE_CONTROL,
            "system": system_cached,
            "messages": msgs,
        }
        probe(f"C. auto+system, {n_turns} turns ({len(msgs)} msgs)", body, results)

    # Repeat the 10-turn to confirm it cached
    msgs_10 = make_turns(10) + [{"role": "user", "content": "Summarize."}]
    body_10 = {
        "model": MODEL, "max_tokens": 10,
        "cache_control": CACHE_CONTROL,
        "system": system_cached,
        "messages": msgs_10,
    }
    probe("C. 10 turns repeated (warm check)", body_10, results)

    print()
    print("=" * 80)
    print("EXPERIMENT D: 4 breakpoints + auto = error?")
    print("=" * 80)

    # Try to exceed 4 breakpoints
    msgs_4bp = list(make_turns(10))
    for idx in [3, 7, 11, 15]:  # 4 message breakpoints
        msgs_4bp[idx] = {
            "role": msgs_4bp[idx]["role"],
            "content": [{
                "type": "text",
                "text": msgs_4bp[idx]["content"],
                "cache_control": CACHE_CONTROL,
            }]
        }
    msgs_4bp.append({"role": "user", "content": "Summarize."})

    # 4 explicit on messages + 1 on system = 5. Should this error?
    body_d1 = {
        "model": MODEL, "max_tokens": 10,
        "system": system_cached,  # 1 breakpoint
        "messages": msgs_4bp,     # 4 breakpoints
    }
    probe("D1. 5 breakpoints (system + 4 msgs)", body_d1, results)

    # 4 explicit on messages + system + auto = 6?
    body_d2 = {
        "model": MODEL, "max_tokens": 10,
        "cache_control": CACHE_CONTROL,
        "system": system_cached,
        "messages": msgs_4bp,
    }
    probe("D2. 6 breakpoints (auto + system + 4 msgs)", body_d2, results)

    # Summary
    print("\n" + "=" * 100)
    print(f"{'Probe':<55} {'Hit?':>4} {'Eff':>8} {'Read':>8} {'Create':>8} {'Input':>7} {'Read%':>7}")
    print("-" * 100)
    for r in results:
        hit = "HIT" if r["read"] > 0 else "MISS"
        print(f"{r['label']:<55} {hit:>4} {r['effective']:>8,} {r['read']:>8,} "
              f"{r['create']:>8,} {r['input']:>7,} {r['read_pct']:>6.1f}%")


if __name__ == "__main__":
    print("KV Cache Breakpoint Behavior Experiment")
    print(f"Model: {MODEL}")
    print()
    run()
