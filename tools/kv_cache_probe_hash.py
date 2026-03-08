"""Probe whether Anthropic's cache verifies full content or hashes boundaries.

If the cache hashes only the start/end of each segment, then mutations
in the MIDDLE of a cached segment will still hit cache (stale KV state).
Mutations at the START or END of a segment would miss.

We test this by making increasingly aggressive mutations at different
positions within a cached segment:
  - Start of segment (first few tokens after previous breakpoint)
  - Middle of segment (deep inside)
  - End of segment (just before the cache_control breakpoint)
  - Wholesale replacement of middle content
  - Length-changing mutations (add/remove significant content)

Usage:
    python tools/kv_cache_probe_hash.py

Requires ANTHROPIC_API_KEY in environment.
"""

import os
import sys
import time
import json
import copy
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


# --- Conversation structure ---
# System [CACHED] → 4 messages → last message [CACHED] → final query
#
# The 4 messages between system and last cached message are the test zone.
# We mutate at different positions within this zone.

SYSTEM_TEXT = (
    "You are a helpful assistant specializing in distributed systems, "
    "operating systems, and computer architecture. You have deep knowledge "
    "of cache coherence protocols, virtual memory management, demand paging, "
    "and memory hierarchy optimization. You provide detailed technical "
    "answers grounded in systems research. "
) * 12  # well above 1024 tokens

MSG_1_USER = (
    "ALPHA_START Explain how modern CPUs implement speculative execution "
    "and how it interacts with the memory hierarchy. Consider both the "
    "performance benefits and the security implications revealed by "
    "Spectre and Meltdown. Discuss the role of the reorder buffer, "
    "branch prediction, and how speculative loads interact with the "
    "cache hierarchy. ALPHA_END"
)

MSG_2_ASSISTANT = (
    "BRAVO_START Speculative execution allows CPUs to execute instructions "
    "before knowing whether they will be needed. The reorder buffer tracks "
    "speculative state and rolls back on misprediction. The key performance "
    "benefit is hiding memory latency — while waiting for a cache miss, "
    "the CPU can speculatively execute hundreds of instructions. "
    "BRAVO_MIDDLE "
    "The security implications are profound. Spectre exploits the fact "
    "that speculative loads bring data into the cache even when rolled "
    "back. An attacker can use timing side channels to read the cached "
    "data, effectively bypassing all software isolation boundaries. "
    "Meltdown specifically exploits speculative reads that cross the "
    "kernel/user boundary. Mitigations like KPTI and retpoline add "
    "significant performance overhead. "
    "BRAVO_END"
)

MSG_3_USER = (
    "CHARLIE_START How does the TLB interact with context switches? "
    "When the OS switches between processes, what happens to the TLB "
    "entries and how does this affect performance? Discuss both full "
    "TLB flushes and the use of ASIDs. CHARLIE_END"
)

# This message has cache_control — it's the end of the cached prefix
MSG_4_ASSISTANT = (
    "DELTA_START TLB management during context switches is one of the "
    "most significant performance factors in modern operating systems. "
    "A full TLB flush on every context switch means the new process "
    "starts with a completely cold TLB, causing page table walks for "
    "every memory access until the TLB is warm again. This can cost "
    "thousands of cycles. ASIDs (Address Space Identifiers) allow "
    "the TLB to hold entries from multiple address spaces simultaneously, "
    "avoiding flushes entirely. However, ASID space is limited — "
    "typical hardware provides 8-16 bits, giving 256-65536 ASIDs. "
    "When ASIDs are exhausted, the OS must flush and reassign. "
    "DELTA_END"
)

FINAL_QUERY = "Summarize the key points in one sentence."


def build_request(system_text, msg1, msg2, msg3, msg4, final):
    """Build an API request body."""
    return {
        "model": MODEL,
        "max_tokens": 10,
        "system": [{"type": "text", "text": system_text, "cache_control": CACHE_CONTROL}],
        "messages": [
            {"role": "user", "content": msg1},
            {"role": "assistant", "content": msg2},
            {"role": "user", "content": msg3},
            {"role": "assistant", "content": [
                {"type": "text", "text": msg4, "cache_control": CACHE_CONTROL}
            ]},
            {"role": "user", "content": final},
        ],
    }


def send(body, label):
    """Send request, return usage stats."""
    resp = httpx.post(URL, headers=HEADERS, json=body, timeout=60)
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
    results = []
    delay = 2

    def probe(label, body):
        r = send(body, label)
        if r:
            results.append(r)
            print(f"  {r['label']:<55} "
                  f"read={r['read']:>6,}  create={r['create']:>6,}  "
                  f"input={r['input']:>5,}  ({r['read_pct']:>5.1f}%)")
        else:
            print(f"  {label:<55} FAILED")
        time.sleep(delay)

    base = build_request(SYSTEM_TEXT, MSG_1_USER, MSG_2_ASSISTANT,
                         MSG_3_USER, MSG_4_ASSISTANT, FINAL_QUERY)

    # --- Phase 1: Establish cache ---
    probe("1a. cold start", base)
    probe("1b. warm (confirm cache)", base)

    # --- Phase 2: Mutate START of msg 1 (beginning of uncached-between zone) ---
    probe("2a. msg1 start: ALPHA_START→ALPHA_XTART",
          build_request(SYSTEM_TEXT,
                        MSG_1_USER.replace("ALPHA_START", "ALPHA_XTART", 1),
                        MSG_2_ASSISTANT, MSG_3_USER, MSG_4_ASSISTANT, FINAL_QUERY))

    # --- Phase 3: Mutate MIDDLE of msg 2 (deep in the zone) ---
    probe("3a. msg2 middle: BRAVO_MIDDLE→BRAVO_XIDDLE",
          build_request(SYSTEM_TEXT, MSG_1_USER,
                        MSG_2_ASSISTANT.replace("BRAVO_MIDDLE", "BRAVO_XIDDLE", 1),
                        MSG_3_USER, MSG_4_ASSISTANT, FINAL_QUERY))

    # --- Phase 4: Mutate END of msg 3 (just before cached breakpoint) ---
    probe("4a. msg3 end: CHARLIE_END→CHARLIE_XND",
          build_request(SYSTEM_TEXT, MSG_1_USER, MSG_2_ASSISTANT,
                        MSG_3_USER.replace("CHARLIE_END", "CHARLIE_XND", 1),
                        MSG_4_ASSISTANT, FINAL_QUERY))

    # --- Phase 5: Mutate START of cached msg 4 ---
    probe("5a. msg4 start: DELTA_START→DELTA_XTART",
          build_request(SYSTEM_TEXT, MSG_1_USER, MSG_2_ASSISTANT,
                        MSG_3_USER,
                        MSG_4_ASSISTANT.replace("DELTA_START", "DELTA_XTART", 1),
                        FINAL_QUERY))

    # --- Phase 6: Mutate END of cached msg 4 ---
    probe("6a. msg4 end: DELTA_END→DELTA_XND",
          build_request(SYSTEM_TEXT, MSG_1_USER, MSG_2_ASSISTANT,
                        MSG_3_USER,
                        MSG_4_ASSISTANT.replace("DELTA_END", "DELTA_XND", 1),
                        FINAL_QUERY))

    # --- Phase 7: LARGE mutation — replace entire middle of msg 2 ---
    gutted_msg2 = (
        "BRAVO_START This content has been completely replaced with "
        "entirely different text that bears no resemblance to the "
        "original. The speculative execution discussion is gone. "
        "Instead here is a recipe for chocolate cake. Mix flour, "
        "sugar, cocoa powder, baking soda, and salt. Add eggs, "
        "buttermilk, oil, and vanilla. Bake at 350F for 30 minutes. "
        "BRAVO_END"
    )
    probe("7a. msg2 wholesale replacement (same markers)",
          build_request(SYSTEM_TEXT, MSG_1_USER, gutted_msg2,
                        MSG_3_USER, MSG_4_ASSISTANT, FINAL_QUERY))

    # --- Phase 8: Length-changing mutation — add substantial content ---
    bloated_msg2 = MSG_2_ASSISTANT + " EXTRA " * 200
    probe("8a. msg2 +200 tokens appended",
          build_request(SYSTEM_TEXT, MSG_1_USER, bloated_msg2,
                        MSG_3_USER, MSG_4_ASSISTANT, FINAL_QUERY))

    # --- Phase 9: Length-changing — remove substantial content ---
    short_msg2 = "BRAVO_START Short answer. BRAVO_END"
    probe("9a. msg2 truncated to ~5 tokens",
          build_request(SYSTEM_TEXT, MSG_1_USER, short_msg2,
                        MSG_3_USER, MSG_4_ASSISTANT, FINAL_QUERY))

    # --- Phase 10: Restore original ---
    probe("10a. original restored", base)

    # --- Summary ---
    print("\n" + "=" * 105)
    print(f"{'Probe':<55} {'Eff':>7} {'Read':>7} {'Create':>7} {'Input':>7} {'Read%':>7}")
    print("-" * 105)
    for r in results:
        # Flag unexpected results
        flag = ""
        if "cold" not in r["label"] and "confirm" not in r["label"]:
            if r["read"] > 0 and "mutated" in r["label"] or "replace" in r["label"] or "trunc" in r["label"] or "append" in r["label"]:
                flag = " ← CACHE HIT DESPITE MUTATION"
            elif r["read"] == 0 and "restored" in r["label"]:
                flag = " ← CACHE MISS ON RESTORE"
        print(f"{r['label']:<55} {r['effective']:>7,} {r['read']:>7,} "
              f"{r['create']:>7,} {r['input']:>7,} {r['read_pct']:>6.1f}%{flag}")


if __name__ == "__main__":
    print("KV Cache Hash Boundary Experiment")
    print(f"Model: {MODEL}")
    print(f"Cached segments: system prompt + through msg 4")
    print(f"Test zone: messages 1-3 (between system and msg4 breakpoints)")
    print(f"Markers: ALPHA/BRAVO/CHARLIE/DELTA with START/MIDDLE/END tags")
    print()
    run()
