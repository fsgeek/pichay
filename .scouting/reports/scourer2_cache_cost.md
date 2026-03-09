Now I have a comprehensive picture. Here's the full report:

---

# Scourer 2: Cache/Cost Behavior Report

## Scope Read

| File | Lines | Read |
|------|-------|------|
| `src/pichay/gateway.py` | 1082 | Full |
| `src/pichay/message_ops.py` | 534 | Full |
| `src/pichay/telemetry.py` | 250 | Full |
| `src/pichay/message_store.py` | 284 | Full |
| `docs/design-cache-aware.md` | 239 | Full |
| `tools/kv_cache_probe.py` | 343 | Full |
| `tools/kv_cache_probe_hash.py` | 253 | Full |
| `tools/kv_cache_probe_lookback.py` | 230 | Full |
| `tools/kv_cache_probe_breakpoints.py` | 244 | Full |
| `tools/analyze_claude_code_cache.py` | 234 | Full |
| `src/pichay/config.py` | 177 | Full |

Also grepped all `src/pichay/` for `cache_control`, `cache_read`, `cache_miss`, `KV cache`.

---

## Key Findings (severity-ordered)

### 1. CRITICAL — `_strip_cache_control` removes ALL client cache breakpoints

**Severity: High** — The MessageStore strips every `cache_control` marker from incoming messages (`message_store.py:41-53`). Claude Code typically sets 2-3 explicit breakpoints (documented at `design-cache-aware.md:144-145`). After stripping, the forwarded request has **zero** explicit breakpoints on messages. The system prompt injection in `message_ops.py` also doesn't re-add any breakpoints.

The design doc says "Pichay gets at most 2 more" slots (`design-cache-aware.md:44`), but right now Pichay uses **none** of them. The forwarded request relies entirely on whatever auto-caching Anthropic applies, which means the carefully placed Claude Code breakpoints (on system blocks and deep conversation messages) are lost.

**Impact**: Cache hit rate drops from achievable ~92% to observed ~44% baseline. At Opus pricing ($5/M base, $0.50/M read), the cost difference over a 100K-token session with 20 turns is enormous — roughly 10x more expensive per cache-missed prefix.

### 2. HIGH — Dynamic anchor mutates last user message every request

**Severity: High** — `inject_system_status` (`message_ops.py:249-388`) appends a `[pichay-live-status]` block with timestamp, token count, and pressure info to the **last user message** on every request. While the design doc correctly identifies that "mutations after the last breakpoint are free" (`design-cache-aware.md:36`), this only holds if breakpoints exist on earlier messages. Since finding #1 strips all message breakpoints, this dynamic content may be within the cached prefix if auto-caching places a breakpoint after the last user message.

The `_strip_stale_status` function (`gateway.py:566-584`) removes old status blocks from prior user messages — but only from the gateway's own message chain. Each prior turn's status was already baked in and forwarded.

### 3. HIGH — `track_usage` counts cache tokens as "effective" but doesn't distinguish cost

**Severity: Medium-High** — `Session.track_usage` (`gateway.py:427-435`) sums `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` into a single "effective" number. This is used for token cap enforcement and pressure calculation. But the cost profile differs radically:
- Cache read: $0.50/M (10x cheaper than base)
- Cache create: $6.25-10/M (1.25-2x more expensive than base)
- Base: $5/M

A session at 150K effective tokens with 90% cache read costs ~$0.83, but at 0% cache read costs ~$0.95 (cache create at 1.25x). The token cap doesn't account for this — it treats a cache-efficient session identically to a thrashing one.

### 4. MEDIUM — Unexpected cache miss detection has blind spots

**Severity: Medium** — `telemetry.py:207-218` flags `cache_miss_unexpected` when `cache_read == 0 and cache_create > 0 and effective > 4096` (skipping the first request). But it doesn't fire when:
- `cache_create == 0` and `input_tokens` is high (no caching attempted at all — which happens when all breakpoints are stripped per finding #1)
- The session has only 1 request (cold start is ignored, correct)

More importantly, it doesn't correlate misses with the *cause* — was it a system prompt mutation, a message mutation, or breakpoint removal? The `cache_miss_unexpected` event lacks the `shrink_ratio` or `duplication_score` that would help diagnosis.

### 5. MEDIUM — `_inspect_sse_chunk` accumulates usage via `.update()` which silently overwrites

**Severity: Medium** — In `gateway.py:149`, `usage_accumulator.update(usage)` overwrites keys. If `message_start` provides `input_tokens: 5000` and `message_delta` later provides `input_tokens: 0`, the final value is `0`. The Anthropic streaming API sends `input_tokens` in `message_start.message.usage` and `output_tokens` in `message_delta.usage` — these shouldn't conflict. But if the API ever sends overlapping keys across events, the later value silently wins. The code is fragile to API format evolution.

### 6. LOW — Design doc's "20-block lookback" was experimentally disproven but code doesn't reflect it

**Severity: Low** — `design-cache-aware.md:137-138` documents that the 20-block lookback hypothesis was disproven: "Full content verification across the entire cached prefix. No free mutations." But the design doc still has a section titled "20-Block Lookback Exploitation" (`design-cache-aware.md:123-131`) that suggests exploiting it. This section should be marked as invalidated to prevent future developers from building on a false premise.

### 7. LOW — `_duplication_score` doesn't account for cache-aware deduplication

**Severity: Low** — `gateway.py:81-86` computes a simple text-equality duplication score across all blocks. This metric doesn't distinguish between *intentional* duplication (system prompt repeated for caching) and *accidental* duplication (same tool result appearing twice). The `outgoing_growth_suspicious` anomaly at `telemetry.py:190-203` uses a 5% threshold that could be tripped by legitimate Pichay injection (yuyay manifests, status anchors).

---

## Evidence (file:line)

| Finding | Primary Evidence |
|---------|-----------------|
| #1 cache_control stripping | `message_store.py:41-53`, `message_store.py:252-256` |
| #2 dynamic anchor mutation | `message_ops.py:296-303`, `gateway.py:566-584` |
| #3 effective token conflation | `gateway.py:427-435`, `config.py:32-33` |
| #4 cache miss blind spots | `telemetry.py:207-218` |
| #5 SSE usage overwrite | `gateway.py:142-153` |
| #6 stale lookback docs | `design-cache-aware.md:123-131` vs `design-cache-aware.md:134-138` |
| #7 duplication score | `gateway.py:81-86`, `telemetry.py:190-203` |
| Design intent (static system prompt) | `message_ops.py:203-246`, `design-cache-aware.md:192-196` |
| Experimental evidence | `tools/kv_cache_probe_lookback.py:181`, `tools/kv_cache_probe_hash.py:146-253` |
| Cache pricing model | `design-cache-aware.md:48` |
| Real-world hit rate data | `design-cache-aware.md:165-167` |

---

## Repro Commands

```bash
# Run the cache probes (requires ANTHROPIC_API_KEY)
python tools/kv_cache_probe.py
python tools/kv_cache_probe_hash.py
python tools/kv_cache_probe_breakpoints.py
python tools/kv_cache_probe_lookback.py

# Analyze existing logs for cache behavior
python tools/analyze_claude_code_cache.py logs/

# Start gateway and observe cache hit rate in stderr
python -m pichay.gateway --claude --log-dir logs/
# Watch for lines like: [sid] [Turn N] 150,000 tok | cache 44%

# Verify cache_control stripping — grep forwarded payloads
# (requires a running session with logging)
grep cache_control logs/gateway_*.jsonl | head -20
```

---

## Confidence

| Finding | Confidence | Basis |
|---------|-----------|-------|
| #1 cache_control stripping | **Very High** | Direct code reading, unambiguous `pop("cache_control")` |
| #2 dynamic anchor | **High** | Code is clear; impact depends on auto-cache breakpoint placement |
| #3 cost conflation | **High** | Arithmetic is straightforward; cost impact depends on usage pattern |
| #4 miss detection gaps | **High** | Logic paths are clear from code |
| #5 SSE overwrite | **Medium** | Depends on Anthropic's actual SSE event schema; current schema appears safe |
| #6 stale docs | **Very High** | Doc explicitly says "NO" but keeps the exploitation section |
| #7 duplication score | **Medium** | Threshold tuning is contextual |

---

## Follow-on Mission Recommendation

- **recommend_follow_on**: yes
- **expected_value**: high
- **suggested_scope**: "Cache Breakpoint Restoration Strategy" — Design and implement the re-injection of cache_control breakpoints into forwarded requests. Focus files: `message_store.py` (where stripping happens), `message_ops.py` (where system prompt and anchor are injected), `gateway.py:_preprocess`. The key question is: should Pichay re-add Claude Code's original breakpoints, or place its own at the stable/mutable boundary it controls? The design doc (`design-cache-aware.md:209-211`) explicitly calls out this affordance: "strip Claude Code's, add Pichay's. All 4 slots available." This is the single highest-impact intervention for reducing API cost — moving from 44% to 92% cache hit rate at Opus pricing is roughly a 5-8x cost reduction on input tokens.
- **stop_condition**: When the gateway forwards requests with at least 2 explicit `cache_control` breakpoints (1 on system prompt, 1 on the last stable message before the dynamic tail), and telemetry shows `cache_read_pct > 80%` sustained across a 10+ turn session.
