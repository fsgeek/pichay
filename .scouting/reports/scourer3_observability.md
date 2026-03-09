I now have enough data to produce the audit. Here's the report:

---

# Scourer 3 — Observability Gaps Audit

## Scope Read

| File | Lines | Status |
|---|---|---|
| `src/pichay/telemetry.py` | 250 | Read in full |
| `src/pichay/gateway.py` | 1082 | Read in full |
| `tests/test_telemetry_contract.py` | 269 | Read in full |
| `tests/test_gateway_chaos.py` | 253 | Read in full |
| `tests/test_gateway_core.py` | 630 | Read in full |
| `tools/dashboard.py` | 240 | Read in full |
| `src/pichay/core/pipeline.py` | Spot-checked emit_event calls |

---

## Key Findings (severity-ordered)

### 1. **HIGH — `usage_accumulator.update()` silently overwrites keys instead of accumulating**

`_inspect_sse_chunk` calls `usage_accumulator.update(usage)` for both `message_start` and `message_delta` events. The Anthropic SSE protocol sends `input_tokens` in `message_start.message.usage` and `output_tokens` in `message_delta.usage`. Because `dict.update()` overwrites keys, if `message_delta` also includes `input_tokens: 0` (which it sometimes does), the accumulated `cache_read_input_tokens` or `input_tokens` from `message_start` will be silently zeroed out. This corrupts the usage dict that feeds `record_request()`, `_check_token_cap()`, and `_display_turn_status()`.

**Impact**: Cache hit rate telemetry, token cap enforcement, and `cache_miss_unexpected` anomaly detection all depend on accurate usage accumulation. A zeroed `cache_read_input_tokens` triggers false `CACHE_MISS_EVENTS` Prometheus increments and spurious `cache_miss_unexpected` events.

### 2. **HIGH — `cache_miss_unexpected` detection uses stale `s.request_count` (race with lock)**

In `telemetry.py:207`, the cache miss check reads `s.request_count` after releasing the lock at line 148. Between the lock release and the check, another thread could increment `request_count`, making a first-request (cold start) incorrectly appear to be a subsequent request. The check `s.request_count > 1` should use a locally captured value inside the lock scope.

### 3. **MEDIUM — No test coverage for `cache_miss_unexpected` event or `CACHE_MISS_EVENTS` counter**

Grep across all test files shows zero tests for the `cache_miss_unexpected` event type or the `CACHE_MISS_EVENTS` Prometheus counter. This is the only anomaly detection path with zero coverage. If the logic breaks (e.g., via Finding #1), nothing catches it.

### 4. **MEDIUM — Dashboard (`_dashboard_html`) and `tools/dashboard.py` are completely separate systems reading different data formats**

The inline `_dashboard_html()` (gateway.py:165-376) fetches from `/health`, `/api/sessions`, `/api/events` — the live gateway API. The standalone `tools/dashboard.py` reads legacy proxy JSONL with `type: "compaction"` and `type: "request"` record formats. The gateway now emits `type: "request_metrics"`, `type: "response_observed"`, etc. — **`tools/dashboard.py` will parse zero records from current gateway logs** because it only matches `type == "compaction"` and `type == "request"`.

### 5. **MEDIUM — Prometheus counter cardinality is unbounded for error labels**

`POLICY_CONFLICTS` uses labels `["winner_stage", "loser_stage"]`. The `emit()` method falls back to `"unknown"` for missing fields, but if callers pass arbitrary stage names, the cardinality grows unbounded. Currently the pipeline only uses `phantom`/`paging`/`trim`, but there's no validation. More critically, `REQ_TOTAL` uses `status` as a string label — every unique HTTP status code creates a new time series.

### 6. **MEDIUM — `response_observed` event is never tested for schema correctness**

Tests check for `request_metrics` contract fields but never assert the shape of `response_observed` events. The `response_observed` event includes `usage`, `response_bytes`, `chunk_count`, `endpoint` — none validated by contract tests. A regression could silently break downstream consumers.

### 7. **LOW — Dashboard KPI "Stream Errors" counts `stream_error` events but not `provider_error`**

The inline dashboard JS (gateway.py:339) filters for `e.type === 'stream_error'` for the "Stream Errors" KPI, but non-streaming provider failures emit `type: "provider_error"` (gateway.py:880-886). These are invisible on the dashboard despite being connectivity failures.

### 8. **LOW — `Telemetry.sessions` uses `defaultdict(SessionSummary)` — querying unknown session IDs creates ghost entries**

`self.sessions` is a `defaultdict(SessionSummary)` (telemetry.py:76). If any code path reads `self.sessions[unknown_id]`, it silently creates an empty entry. Currently no code path does this incorrectly, but the `session_summary()` endpoint would expose phantom sessions if a typo or dash-board JS bug requested one.

### 9. **LOW — Telemetry log file is opened/closed on every `emit()` call**

`emit()` (telemetry.py:116) opens the log file in append mode for every single event, under the lock. At high request rates with many pipeline events per request, this creates significant I/O overhead and lock contention. Not a correctness bug but an observability bottleneck under load.

---

## Evidence

| Finding | File:Line | Detail |
|---|---|---|
| #1 | `gateway.py:149,153` | `usage_accumulator.update(usage)` — overwrites, not merges |
| #2 | `telemetry.py:144-148` vs `telemetry.py:207` | Lock released at 148, `s.request_count` read at 207 outside lock |
| #3 | All test files | `grep cache_miss` returns zero matches |
| #4 | `tools/dashboard.py:38,55` | Matches only `type == "compaction"` and `type == "request"` |
| #5 | `telemetry.py:14-18` | `REQ_TOTAL` labels include `status` as string |
| #6 | `test_telemetry_contract.py` | No `response_observed` in required field sets |
| #7 | `gateway.py:339` | `ev.filter(e => e.type === 'stream_error')` — excludes `provider_error` |
| #8 | `telemetry.py:76` | `defaultdict(SessionSummary)` |
| #9 | `telemetry.py:116-117` | `open()` inside `emit()` per call |

---

## Repro Commands

```bash
# Run existing tests to confirm current state
cd /home/tony/projects/pichay
uv run pytest tests/test_telemetry_contract.py tests/test_gateway_chaos.py tests/test_gateway_core.py -v

# Confirm zero cache_miss test coverage
grep -r "cache_miss" tests/

# Confirm tools/dashboard.py won't parse gateway logs
grep -c '"type": "request_metrics"' logs/gateway_*.jsonl 2>/dev/null  # these exist
grep -c '"type": "request"' logs/gateway_*.jsonl 2>/dev/null          # these don't

# Repro Finding #1: send a stream with message_delta containing input_tokens:0
# (requires live gateway or extended fake client test)
```

---

## Confidence

**High** for findings #1, #3, #4, #7 — directly observable from code. **Medium** for #2 — requires concurrent requests to manifest. **Medium** for #5 — bounded in practice by the small set of HTTP status codes and pipeline stages but architecturally concerning.

---

## Follow-on Mission Recommendation

- **recommend_follow_on**: yes
- **expected_value**: high
- **suggested_scope**: Fix the `usage_accumulator` merge logic in `_inspect_sse_chunk` (use additive merge, not `dict.update`); add test coverage for `cache_miss_unexpected`; add `response_observed` contract test; retire or update `tools/dashboard.py` to read current event format; surface `provider_error` in dashboard KPIs.
- **stop_condition**: All Prometheus counters tested for correctness under streaming usage scenarios; `tools/dashboard.py` either deleted or updated to parse `request_metrics` events; `response_observed` schema has a contract test.
