"""Probe the 20-block lookback window.

Builds a conversation with many blocks, places a cache breakpoint
at the end, then mutates at increasing distances from the breakpoint
to find where mutations become invisible to cache verification.

Structure:
    System prompt [CACHED]
    40 message pairs (80 messages total)
    Final cached assistant message [CACHED breakpoint]
    Final user query (uncached)

We mutate at distances 1, 5, 10, 15, 20, 25, 30, 35 messages
from the breakpoint and measure cache behavior.

Usage:
    python tools/kv_cache_probe_lookback.py

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

# System prompt — large enough to cache on Sonnet (>2048 tokens)
SYSTEM_TEXT = (
    "You are a helpful assistant specializing in distributed systems. "
    "You provide concise, accurate answers about cache coherence, "
    "virtual memory, demand paging, and memory hierarchy optimization. "
) * 20

# Generate 40 turn pairs (80 messages). Each pair is short but distinct.
# We use numbered markers so we can mutate specific messages precisely.
def make_messages():
    """Build 80 messages (40 turns) plus a final cached message and query."""
    msgs = []
    for i in range(40):
        msgs.append({
            "role": "user",
            "content": f"TURN_{i:03d}_USER: Explain concept number {i} "
                       f"in distributed systems memory management."
        })
        msgs.append({
            "role": "assistant",
            "content": f"TURN_{i:03d}_ASSISTANT: Concept {i} relates to "
                       f"how distributed systems handle memory state across "
                       f"multiple nodes. The key trade-off at level {i} is "
                       f"between consistency and availability."
        })

    # Final assistant message with cache breakpoint
    msgs.append({
        "role": "user",
        "content": "FINAL_USER: Summarize all concepts."
    })
    msgs.append({
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": "FINAL_ASSISTANT: All 40 concepts relate to the fundamental "
                    "tension between consistency, availability, and partition "
                    "tolerance in distributed memory systems.",
            "cache_control": CACHE_CONTROL,
        }],
    })

    # Uncached query
    msgs.append({
        "role": "user",
        "content": "Give a one-word summary."
    })

    return msgs


def mutate_at_distance(messages, distance_from_end):
    """Mutate a message at the given distance from the cache breakpoint.

    Distance is in messages (not turns). The cache breakpoint is on
    the second-to-last message (index -2). So distance 1 is the message
    just before it, distance 2 is two before, etc.

    We mutate by changing one word to include an X.
    """
    msgs = []
    for m in messages:
        d = dict(m)
        if isinstance(d.get("content"), list):
            d["content"] = [dict(b) for b in d["content"]]
        msgs.append(d)

    # The cached breakpoint is at index -2 (final assistant message)
    # Distance 1 = index -3, distance 2 = index -4, etc.
    target_idx = len(msgs) - 2 - distance_from_end

    if target_idx < 0:
        return None, -1

    content = msgs[target_idx].get("content", "")
    if isinstance(content, str):
        # Replace TURN_XXX with XURN_XXX
        msgs[target_idx]["content"] = content.replace("TURN_", "XURN_", 1)
        if content.startswith("FINAL"):
            msgs[target_idx]["content"] = content.replace("FINAL_", "XINAL_", 1)
    elif isinstance(content, list):
        block = content[0]
        block["text"] = block["text"].replace("FINAL_", "XINAL_", 1)

    return msgs, target_idx


def send(system, messages, label):
    """Send request, return usage stats."""
    body = {
        "model": MODEL,
        "max_tokens": 10,
        "system": [{"type": "text", "text": system, "cache_control": CACHE_CONTROL}],
        "messages": messages,
    }
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


def run():
    base_msgs = make_messages()
    results = []
    delay = 2

    print(f"Total messages: {len(base_msgs)}")
    print(f"Cache breakpoint at message index: {len(base_msgs) - 2}")
    print()

    def probe(label, msgs):
        r = send(SYSTEM_TEXT, msgs, label)
        if r:
            results.append(r)
            hit = "HIT" if r["read"] > 0 else "MISS"
            print(f"  {r['label']:<50} {hit:>4}  "
                  f"read={r['read']:>6,}  create={r['create']:>6,}  "
                  f"input={r['input']:>5,}  ({r['read_pct']:>5.1f}%)")
        else:
            print(f"  {label:<50} FAILED")
        time.sleep(delay)

    # Establish cache
    probe("baseline (cold)", base_msgs)
    probe("baseline (warm)", base_msgs)

    # Mutate at increasing distances from breakpoint
    for dist in [1, 2, 3, 5, 10, 15, 18, 19, 20, 21, 22, 25, 30, 35, 40, 50, 60, 79]:
        mutated, idx = mutate_at_distance(base_msgs, dist)
        if mutated is None:
            print(f"  distance {dist}: out of range, skipping")
            continue
        probe(f"mutate distance={dist} (msg idx {idx})", mutated)

    # Restore original
    probe("original restored", base_msgs)

    # Summary
    print("\n" + "=" * 100)
    print(f"{'Probe':<50} {'Hit?':>4} {'Eff':>8} {'Read':>8} {'Create':>8} {'Input':>7} {'Read%':>7}")
    print("-" * 100)
    for r in results:
        hit = "HIT" if r["read"] > 0 else "MISS"
        print(f"{r['label']:<50} {hit:>4} {r['effective']:>8,} {r['read']:>8,} "
              f"{r['create']:>8,} {r['input']:>7,} {r['read_pct']:>6.1f}%")

    # Find the boundary
    print("\n" + "=" * 100)
    print("LOOKBACK BOUNDARY ANALYSIS:")
    last_miss = None
    first_hit_after_miss = None
    for r in results:
        if "distance=" in r["label"]:
            dist = int(r["label"].split("distance=")[1].split(" ")[0])
            if r["read"] == 0:
                last_miss = dist
            elif last_miss is not None and first_hit_after_miss is None:
                first_hit_after_miss = dist

    if first_hit_after_miss:
        print(f"  Last miss at distance: {last_miss}")
        print(f"  First hit at distance: {first_hit_after_miss}")
        print(f"  Lookback window appears to be: {first_hit_after_miss - 1} to {last_miss} blocks")
    elif last_miss:
        print(f"  All mutations caused misses up to distance {last_miss}")
        print(f"  Lookback window may be larger than tested range")
    else:
        print(f"  No misses detected — all mutations were invisible to cache")


if __name__ == "__main__":
    print("KV Cache 20-Block Lookback Experiment")
    print(f"Model: {MODEL}")
    print(f"Structure: system[C] → 80 messages (40 turns) → final[C] → query")
    print()
    run()
