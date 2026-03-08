# Cache-Aware Pichay: Design Constraints and Direction

Date: 2026-03-07
Status: Constraints captured. Design not finalized. Key questions remain.

## Origin

Tony noticed he was hitting Anthropic rate limits for the first time despite
running less than before Pichay. Investigation revealed Pichay was thrashing
Anthropic's KV cache by mutating the message prefix on every request
(timestamp, token count, eviction changes). The "optimization" was increasing
actual compute cost while decreasing token count — the metrics said it was
working while the bill said otherwise.

## What We Proved (Experimentally)

Probe scripts at `tools/kv_cache_probe.py` and `tools/kv_cache_probe_hash.py`.

1. **Cache is prefix-sequential.** Breakpoints define how far the cached prefix
   extends. Everything before the deepest breakpoint is one cached unit.

2. **Full content verification within lookback window.** One token changed
   anywhere within 20 blocks of a breakpoint = complete cache miss.

3. **20-block lookback limit.** Mutations more than 20 blocks before a
   breakpoint are NOT checked. The cache may hit despite content changes
   in that region. (Needs further experimental confirmation.)

4. **Multiple cache entries coexist.** Creating a new cache entry doesn't
   evict the old one. Original prefix survives and can be restored.

5. **System prompt mutation is catastrophic.** Pichay's timestamp/token-count
   injection into the system prompt invalidated the entire cache on every
   single request. This was the primary cause of the rate limit problem.

6. **Mutations after the last breakpoint are free.** Content beyond the
   cached prefix doesn't affect cache behavior.

## Anthropic Cache API Constraints

- **Type**: only `"ephemeral"`.
- **TTL**: 5min default (1.25x write), 1h option (2x write). Maximum, not guaranteed.
- **Max breakpoints**: 4 per request.
- **Claude Code uses ~2** on system prompt blocks. Pichay gets at most 2 more.
- **Minimum cacheable tokens**: Opus 4.6 = 4,096.
- **Automatic caching**: top-level `cache_control` auto-advances breakpoint to last
  cacheable block. Designed for multi-turn conversation growth.
- **Pricing (Opus 4.6)**: base $5/M, write $6.25-$10/M, read $0.50/M.

## Three Operating Regimes

| Regime | Cache | Pichay strategy |
|--------|-------|----------------|
| Direct to Anthropic/OpenAI/Google | High hit rate | Preserve prefix stability. Context surgery in response path. |
| OpenRouter (brokered) | Random/none | Full context surgery on inbound. No cache to thrash. |
| Local (LM Studio, vLLM) | None | Full context surgery. Size = latency. |

Pichay knows the upstream URL and can switch strategy per-regime.

## Architecture: Dual Address Spaces

**The core idea:** Pichay maintains two views of the conversation.

- **Virtual address space** (Claude Code's view): The full, unmodified conversation.
  Claude Code manages its own history, approaching its compaction threshold naturally.
- **Physical address space** (Anthropic's view): A curated, smaller conversation
  that Pichay constructs from Claude Code's messages. Stable prefix, cache-friendly.

Pichay maintains the **page table**: the mapping between Claude Code's messages
and the curated versions Anthropic receives.

### Request Flow (Cached Regime)

```
Claude Code sends:  [m1, m2, m3, m4, m5, m6_new_user]
                          |
Pichay page table:  m1→m1' (curated), m2→m2' (tombstone), m3→m3, m4→m4' (curated)
                          |
Anthropic receives: [m1', m2', m3, m4', m5, m6_new_user]
                          |
Anthropic responds: [r6]  (cache hits on stable prefix m1'..m5)
                          |
Pichay transforms:  r6 → r6' (evict cold content, add tombstones)
                          |
Claude Code stores: r6' in its history
```

### Response-Path Transformation

Pichay can modify the response BEFORE Claude Code sees it:
- Replace verbose tool results with tombstones
- The modified version is what Claude Code stores
- Next request includes the modified version naturally
- Anthropic sees it for the first time, caches it
- Stable from that point on

This means mutations happen ONCE (on the way out) then stabilize.
No repeated mutation of the same content.

### Scavenger Eviction

Pichay doesn't need its own eviction policy for the cached regime:
1. Detect when Claude Code's own eviction drops content (message count drops,
   content disappears between consecutive requests)
2. Catch the dropped content before it's lost
3. Curate it (tensors → Apacheta) or summarize it (cheap model fallback)
4. The cache miss from Claude Code's eviction is the free window for surgery

### Cooperative Curation

At natural transition points (compaction, Claude Code eviction):
1. Give Opus the about-to-die content: "This is leaving context. Curate what
   you want to keep. Write tensors."
2. If Opus curates → high-quality memory formation (T/I/F as functions)
3. If Opus declines → fall back to cheap-model summarization
4. Either way, content doesn't just vanish

**Pressure mechanism:** If the model doesn't curate, content is subject to
external summarization (lower quality). The model that curates its own
memories gets better memories. Aligned incentives.

### 20-Block Lookback Exploitation

If mutations are >20 blocks before the nearest cache breakpoint, the cache
doesn't verify them. This means:
- Old, stable tombstones deep in the conversation can be placed freely
- Only the most recent ~20 blocks need to be truly stable
- Pichay can potentially mutate old content without cache cost
- **NEEDS EXPERIMENTAL CONFIRMATION** — the lookback behavior was documented
  but not yet tested with our probes

## Answered Questions (2026-03-07 experiments)

1. **Does the 20-block lookback allow invisible mutations?**
   **NO.** Tested mutations at distances 1-79 from breakpoint. Every single one
   caused a full cache miss. The "20-block lookback" in the docs refers to
   search depth for finding matching prefixes, not verification depth.
   Full content verification across the entire cached prefix. No free mutations.

2. **How does Claude Code set its cache_control breakpoints?**
   From log analysis (3,475 requests, 55 sessions):
   - 2 explicit breakpoints on system prompt blocks (.system[1], .system[2])
   - 1 breakpoint on a message deep in conversation (varies by session)
   - Uses `{"type": "ephemeral", "ttl": "1h"}` and `{"type": "ephemeral"}`
   - Total: 3 breakpoints typically. Pichay gets 1 more.

3. **Can Pichay add its own cache_control breakpoints?**
   Hard limit is 4 total. 5+ returns 400 error. Claude Code uses 2-3.
   Adding/removing breakpoints changes request structure → cache miss on
   transition, but stabilizes after.

4. **Does automatic caching work alongside explicit breakpoints?**
   **YES.** Top-level `cache_control` is additive with block-level breakpoints.
   Auto cache reads from existing explicit-breakpoint caches (cross-compatible).
   Auto cache uses 1 of the 4 slots.

5. **How does cache behave with growing conversations?**
   Beautifully. Auto caching naturally advances:
   - Turn N: reads prefix from cache, creates new tail
   - Turn N+1: reads larger prefix (including N's tail), creates new tail
   - Mutating only the last message preserves 77%+ cache hit rate
   - This is EXACTLY the append-only pattern Pichay's dual address space needs.

6. **Actual cache hit rate through Pichay?**
   44.4% overall across 55 sessions. Individual sessions ranged from 0% to 92%.
   Sessions where Pichay wasn't mutating aggressively hit 92% — that's the
   achievable baseline. The gap (44% → 92%) is the cost of prefix mutation.

## Remaining Open Questions

1. **How does Claude Code decide when to compact?**
   We can detect it (message count drops). Observed 235 compaction events across
   55 sessions. One session had 15 compactions in 99 requests. Can we influence
   timing? Unknown.

2. **Page table reconciliation at compaction.**
   When Claude Code compacts, Pichay's mappings for compacted messages are
   invalidated. Need a reconciliation strategy. Likely: detect compaction,
   rebuild page table from new message stream, accept one cache miss.

3. **Interaction between Pichay's curated stream and Claude Code's compaction.**
   If Pichay sends a smaller stream to Anthropic, Claude Code doesn't know.
   When Claude Code compacts based on its own (larger) view, the messages it
   references may not match what Anthropic saw. Need to handle gracefully.

8. **How does the page table reconcile when Claude Code's own compaction fires?**
   If Claude Code rewrites messages 1-20 into a summary, Pichay's mappings
   for those messages are invalidated. Need a reconciliation strategy.

## Immediate Next Steps

1. **Stop the bleeding.** Remove dynamic content (timestamp, token count) from
   Pichay's system prompt injection. Make system prompt static. Move dynamic
   info to end-of-messages anchor (after last breakpoint, uncached anyway).
   This is a one-line fix with immediate measurable impact.

2. **Use automatic caching.** Add top-level `cache_control` to Pichay's
   forwarded requests (if Claude Code doesn't already). This gives the
   growing-conversation cache pattern for free.

3. **Build the page table.** Prototype the dual address space: hash each
   message block from Claude Code, maintain a mapping to curated versions,
   forward the curated stream to Anthropic. Append-only on the Anthropic side.

4. **Instrument cache hit rate.** Add per-request cache_read_pct to Pichay's
   stderr status line. Make the invisible visible. Target: >90% sustained.

## Gateway Affordances (unexplored)

Pichay controls the wire. It can rewrite anything before Anthropic sees it:
- **cache_control breakpoints**: strip Claude Code's, add Pichay's. All 4 slots
  available. Place them at the stable/mutable boundary Pichay controls.
- **System prompt**: can be entirely replaced, not just appended to.
- **Tools**: can add, remove, redefine. Phantom tools already do this.
- **Messages**: the dual address space means Anthropic sees Pichay's curated
  stream, not Claude Code's raw stream.
- **Model parameter**: could route different requests to different models.
- **Response**: can transform before Claude Code sees it.

The gateway position allows Pichay to use the transformer in ways Claude Code
doesn't anticipate. What this enables is deliberately left open — premature
collapse here would close off possibilities we can't see yet. Revisit as the
dual address space matures.

## Connection to Broader Project

- **Apacheta**: Curated tensors go to the tensor database. Pichay is the
  mechanism that decides WHEN curation happens (at natural transitions).
  The model decides WHAT to curate.
- **Chasqui/Scouts**: Cheap models via OpenRouter review curated tensors.
  Full context surgery on the cacheless OpenRouter path. Budget recovered
  from cache-aware operation funds the scout fleet.
- **Memory hierarchy**: L1 (live context) → L2 (tensors) transition is
  the curation moment. Pichay provides the infrastructure. The model
  provides the judgment.
- **Compaction vs Curation**: Compaction (summarization) preserves the gist.
  Curation (tensorization) preserves cognitive state as reapplicable functions.
  The model should curate; the framework should compact as fallback.
