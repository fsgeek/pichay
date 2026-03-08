"""Probe Anthropic's KV cache behavior — systematic experiment.

Builds a conversation with multiple cache breakpoints, then
systematically mutates at different positions to measure which
cache segments survive.

Structure:
    System prompt [CACHED]
    Messages 1-4   (early turns)
    Message 5      [CACHED breakpoint]
    Messages 6-8   (late turns)
    Message 9      [CACHED breakpoint]
    Message 10     (final user query, uncached)

Probes mutate at each region boundary and measure cache_read
vs cache_creation to determine:
    - Whether cache segments are independent or prefix-chained
    - Positional sensitivity of mutations
    - Whether multiple cache entries coexist
    - What the cache actually covers (block only? everything before?)

Usage:
    python tools/kv_cache_probe.py

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

# --- Build the conversation structure ---

# System prompt: large enough to be cacheable (>1024 tokens)
SYSTEM_TEXT = (
    "You are a helpful assistant working on a distributed systems project. "
    "The project involves building infrastructure for managing computational "
    "state across multiple processes and machines. Key areas include cache "
    "coherence protocols, memory hierarchy optimization, virtual memory "
    "management, and demand paging systems. The system uses Python for the "
    "control plane and Rust for performance-critical paths. Storage is "
    "handled by ArangoDB for graph queries and DuckDB for analytics. "
    "The architecture follows a layered design with clear separation "
    "between policy and mechanism. "
) * 10  # ~3000+ tokens

# Early messages (turns 1-2, not individually cached)
EARLY_MESSAGES = [
    {"role": "user", "content":
        "Explain the relationship between cache coherence and virtual memory. "
        "How do modern systems maintain consistency across multiple levels of "
        "the memory hierarchy when processes share address spaces? Consider "
        "both hardware and software approaches." * 3},
    {"role": "assistant", "content":
        "Cache coherence and virtual memory interact at several levels. "
        "At the hardware level, MESI/MOESI protocols track cache line state "
        "across cores. At the software level, the TLB shootdown mechanism "
        "ensures page table changes propagate. The key insight is that "
        "coherence is a property of the observation, not the state — what "
        "matters is that all observers agree on the order of writes, not "
        "that they see writes simultaneously. Virtual memory adds a layer "
        "of indirection that both helps (page-level isolation) and hurts "
        "(TLB coherence overhead). Modern systems use a combination of "
        "hardware snooping, directory-based protocols, and software "
        "barriers to maintain consistency across the full hierarchy." * 3},
    {"role": "user", "content":
        "What about the cost model? How do we reason about the tradeoff "
        "between keeping data resident versus paging it out and risking "
        "a fault? The classical analysis assumes uniform page sizes but "
        "our objects vary by orders of magnitude." * 3},
    {"role": "assistant", "content":
        "The cost model for variable-size objects breaks the classical "
        "analysis in several ways. First, the replacement policy can't "
        "treat all objects equally — evicting a 4KB object versus a 400KB "
        "object has very different costs both for the eviction itself and "
        "for the potential fault. Second, the fault cost isn't fixed — it "
        "depends on the current working set size because restoring an object "
        "requires reprocessing the full context. Third, the benefit of "
        "eviction is proportional to the remaining lifetime of the "
        "conversation, which is unknown. The optimal policy would need "
        "to estimate access probability, remaining lifetime, object size, "
        "and current fill level simultaneously." * 3},
]

# Middle messages (turns 3-4, message 5 is the cached breakpoint)
MID_MESSAGES = [
    {"role": "user", "content":
        "How does this compare to the LRU approximation that most operating "
        "systems use? Is there a practical policy that accounts for "
        "variable object sizes without being computationally expensive?" * 3},
    {"role": "assistant", "content":
        "LRU works well for uniform-size pages because recency is a "
        "reasonable proxy for access probability, and eviction cost is "
        "constant. For variable-size objects, you need something like "
        "GreedyDual-Size, which weights recency by object size and cost. "
        "The key modification: instead of evicting the least recently used "
        "object, evict the object with the lowest value-to-size ratio, "
        "where value decays with time since last access. This naturally "
        "keeps large frequently-accessed objects while eagerly evicting "
        "large infrequently-accessed ones." * 3},
]

# Late messages (turns 5-6, after mid cache breakpoint)
LATE_MESSAGES = [
    {"role": "user", "content":
        "What about the interaction between the paging policy and the "
        "provider's own caching infrastructure? If the provider caches "
        "computational state for repeated prefixes, our mutations might "
        "invalidate their cache." * 3},
    {"role": "assistant", "content":
        "This is the critical insight. If the provider implements a KV "
        "cache for repeated prefixes, then our paging system faces a "
        "fundamental tension: every mutation we make to reduce context "
        "size also invalidates the provider's cache. The net effect "
        "depends on the relative costs — cache miss penalty versus "
        "per-token cost of a larger but cache-friendly context. In the "
        "worst case, aggressive paging causes cache thrashing that "
        "increases total cost even while reducing token count." * 3},
]

# Final query (uncached)
FINAL_QUERY = "Given all of this, what is the optimal strategy?"


def build_system():
    """System prompt as a cached block."""
    return [{"type": "text", "text": SYSTEM_TEXT, "cache_control": CACHE_CONTROL}]


def build_messages(
    early=None, mid=None, late=None, final=None,
    cache_after_mid=True, cache_after_late=True,
):
    """Build message array with optional cache breakpoints.

    Cache breakpoints go on the last message before each boundary.
    """
    e = list(early or EARLY_MESSAGES)
    m = list(mid or MID_MESSAGES)
    la = list(late or LATE_MESSAGES)
    f = final or FINAL_QUERY

    msgs = []

    # Early messages (no cache)
    for msg in e:
        msgs.append(dict(msg))

    # Mid messages — last one gets cache breakpoint
    for i, msg in enumerate(m):
        d = dict(msg)
        if cache_after_mid and i == len(m) - 1:
            # Cache breakpoint: wrap content in content block with cache_control
            d["content"] = [
                {"type": "text", "text": msg["content"], "cache_control": CACHE_CONTROL}
            ]
        msgs.append(d)

    # Late messages — last one gets cache breakpoint
    for i, msg in enumerate(la):
        d = dict(msg)
        if cache_after_late and i == len(la) - 1:
            d["content"] = [
                {"type": "text", "text": msg["content"], "cache_control": CACHE_CONTROL}
            ]
        msgs.append(d)

    # Final query (never cached)
    msgs.append({"role": "user", "content": f})

    return msgs


def mutate_message(messages, index, find, replace):
    """Return a copy with one word changed in a specific message."""
    msgs = [dict(m) for m in messages]
    content = msgs[index].get("content", "")
    if isinstance(content, str):
        msgs[index]["content"] = content.replace(find, replace, 1)
    elif isinstance(content, list):
        # Content block format
        blocks = [dict(b) for b in content]
        blocks[0]["text"] = blocks[0]["text"].replace(find, replace, 1)
        msgs[index]["content"] = blocks
    return msgs


def make_request(system, messages, label):
    """Send a request and return usage stats."""
    body = {
        "model": MODEL,
        "max_tokens": 10,
        "system": system,
        "messages": messages,
    }
    resp = httpx.post(URL, headers=HEADERS, json=body, timeout=60)
    if resp.status_code != 200:
        print(f"  ERROR [{label}]: {resp.status_code} {resp.text[:300]}", file=sys.stderr)
        return None
    data = resp.json()
    usage = data.get("usage", {})
    result = {
        "label": label,
        "input_tokens": usage.get("input_tokens", 0),
        "cache_creation": usage.get("cache_creation_input_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    effective = result["input_tokens"] + result["cache_creation"] + result["cache_read"]
    read_pct = (result["cache_read"] / effective * 100) if effective > 0 else 0
    result["effective"] = effective
    result["cache_read_pct"] = round(read_pct, 1)
    return result


def run_experiment():
    system = build_system()
    base_msgs = build_messages()
    results = []
    delay = 2  # seconds between probes

    probes = [
        # --- Phase 1: Establish baseline ---
        ("1a. cold start",
         system, base_msgs),

        ("1b. identical (cache warm)",
         system, base_msgs),

        # --- Phase 2: Mutate system prompt ---
        ("2a. system prompt mutated",
         [{"type": "text", "text": SYSTEM_TEXT.replace("distributed systems", "distributed_systems", 1),
           "cache_control": CACHE_CONTROL}],
         base_msgs),

        ("2b. system restored",
         system, base_msgs),

        # --- Phase 3: Mutate within each region ---
        # Early region (before mid cache breakpoint)
        ("3a. early msg mutated (msg 0)",
         system, mutate_message(base_msgs, 0, "relationship between", "relationship_between")),

        # Mid region (within mid cached block)
        ("3b. mid msg mutated (msg 3, cached)",
         system, mutate_message(base_msgs, 3, "LRU works well", "LRU works_well")),

        # Between mid and late cache breakpoints
        ("3c. late msg mutated (msg 4)",
         system, mutate_message(base_msgs, 4, "interaction between", "interaction_between")),

        # Within late cached block
        ("3d. late cached msg mutated (msg 5, cached)",
         system, mutate_message(base_msgs, 5, "critical insight", "critical_insight")),

        # Final uncached message
        ("3e. final msg mutated",
         system, base_msgs[:-1] + [{"role": "user", "content": FINAL_QUERY + " Be brief."}]),

        # --- Phase 4: Restore and check coexistence ---
        ("4a. original restored",
         system, base_msgs),

        # --- Phase 5: No cache breakpoints on messages ---
        ("5a. no message cache (system only)",
         system, build_messages(cache_after_mid=False, cache_after_late=False)),

        ("5b. no message cache repeated",
         system, build_messages(cache_after_mid=False, cache_after_late=False)),

        # --- Phase 6: Only mid cache, no late ---
        ("6a. mid cache only",
         system, build_messages(cache_after_mid=True, cache_after_late=False)),

        ("6b. mid cache only, late mutated",
         system, mutate_message(
             build_messages(cache_after_mid=True, cache_after_late=False),
             4, "interaction between", "interaction_between")),
    ]

    for label, sys_blocks, msgs in probes:
        r = make_request(sys_blocks, msgs, label)
        if r:
            results.append(r)
            print(f"  {r['label']:<45} read={r['cache_read']:>6,}  "
                  f"create={r['cache_creation']:>6,}  "
                  f"uncached={r['input_tokens']:>5,}  "
                  f"({r['cache_read_pct']:>5.1f}% cached)")
        else:
            print(f"  {label:<45} FAILED")
        time.sleep(delay)

    # --- Summary ---
    print("\n" + "=" * 100)
    print(f"{'Probe':<45} {'Effective':>9} {'Read':>8} {'Create':>8} {'Uncached':>8} {'Read%':>7}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:<45} {r['effective']:>9,} {r['cache_read']:>8,} "
              f"{r['cache_creation']:>8,} {r['input_tokens']:>8,} {r['cache_read_pct']:>6.1f}%")

    # --- Interpretation ---
    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE:")
    print("  - cache_read: tokens loaded from cache (cheap)")
    print("  - cache_creation: tokens written to new cache (expensive)")
    print("  - uncached (input_tokens): tokens processed without caching")
    print()
    print("  If cache is PREFIX-SEQUENTIAL:")
    print("    Mutating early content invalidates all later cache segments")
    print("  If cache is SEGMENTED:")
    print("    Each cache_control block is independent; mutations only")
    print("    invalidate the segment they're in")

    return results


if __name__ == "__main__":
    print("KV Cache Systematic Experiment")
    print(f"Model: {MODEL}")
    print(f"System prompt: ~{len(SYSTEM_TEXT)} chars")
    print(f"Structure: system[C] → 4 early → 2 mid[C] → 2 late[C] → 1 final")
    print(f"  [C] = cache_control breakpoint")
    print()
    run_experiment()
