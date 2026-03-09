Now I have everything. Here's the full report:

---

# Scourer 4 — Provider Compatibility & Catch-All Routing

## Scope Read

| File | Lines | Read |
|------|-------|------|
| `src/pichay/providers/__init__.py` | ~10 | Full |
| `src/pichay/providers/base.py` | ~16 | Full |
| `src/pichay/providers/anthropic.py` | 64 | Full |
| `src/pichay/providers/openai.py` | 77 | Full |
| `src/pichay/gateway.py` | ~1082 | Key sections (routes, preprocessing, catch-all, request handling) |
| `src/pichay/message_ops.py` | Lines 240–300 | System injection logic |
| `src/pichay/core/models.py` | 41 | Full |
| `tests/test_gateway_core.py` | 630 | Via agent |
| `tests/test_gateway_chaos.py` | 422 | Via agent |
| `tests/test_telemetry_contract.py` | 269 | Via agent |

---

## Key Findings (severity-ordered)

### 1. MEDIUM — Catch-all route hardcodes Anthropic upstream, breaks OpenAI clients

The catch-all proxy at `gateway.py:971-993` forwards **all** unhandled `/v1/{path:path}` requests to `https://api.anthropic.com`, regardless of which provider the client expects. An OpenAI-connected client hitting any non-`/v1/chat/completions` route (e.g., `/v1/models`) will get Anthropic's response (likely a 401 or schema mismatch). The forwarded headers (`x-api-key`, `anthropic-version`) are also Anthropic-specific — no `Authorization: Bearer` header is forwarded for OpenAI.

### 2. MEDIUM — Preprocessing skipped for OpenAI requests

`gateway.py:775` gates preprocessing on `endpoint == "messages"`. The OpenAI route passes `"chat_completions"`, so `_preprocess()` is never called for OpenAI traffic. This means:
- No message ingestion/compaction
- No system status injection
- No block labeling
- No page fault detection
- No stale status stripping

OpenAI requests pass through the pipeline (normalize → policy → denormalize) but skip the entire MessageStore layer. This is likely intentional for the paper's scope (Claude Code is the primary client), but it means the paging system is **inert** for OpenAI.

### 3. LOW — OpenAI adapter silently drops `system` field

`CanonicalRequest.system` is populated by the Anthropic adapter (`anthropic.py:38`) but the OpenAI adapter never reads or writes it (`openai.py` contains zero references to "system"). If an OpenAI request somehow had system content set on the canonical request (e.g., by the pipeline), it would be silently discarded during denormalization. Not a bug today since preprocessing doesn't run for OpenAI, but a latent gap if OpenAI support is ever activated.

### 4. LOW — Catch-all creates a new `httpx.AsyncClient` per request

`gateway.py:983` instantiates `httpx.AsyncClient()` inside the request handler with `async with`. This creates and tears down a connection pool on every catch-all request. The main routes use pre-initialized persistent clients (`app.state.clients`). Not a correctness issue but a performance/resource concern under load.

### 5. LOW — No test coverage for catch-all route

Grep for `proxy_passthrough` and `catch.all` in `tests/` returns zero hits. The catch-all route has no test coverage — neither for the happy path (Anthropic passthrough) nor for the failure case (OpenAI client hitting it).

### 6. INFO — OpenAI tool format not translated

Anthropic and OpenAI use different tool schemas. The OpenAI adapter passes `tools` through verbatim (`openai.py:49,71`). If the pipeline modifies tool definitions using Anthropic's schema assumptions, those mutations may produce invalid OpenAI tool definitions.

---

## Evidence

| Finding | File:Line | Detail |
|---------|-----------|--------|
| #1 Hardcoded Anthropic | `gateway.py:980` | `url = f"https://api.anthropic.com/v1/{path}"` |
| #1 Anthropic-only headers | `gateway.py:974-978` | `x-api-key`, `anthropic-version` — no `Authorization: Bearer` |
| #2 Preprocess gate | `gateway.py:775` | `if endpoint == "messages":` |
| #2 OpenAI endpoint name | `gateway.py:963` | `"chat_completions"` |
| #3 No system in OpenAI | `openai.py:1-77` | Zero references to `system` |
| #3 System in Anthropic | `anthropic.py:38,55-56` | `system=payload.get("system")` / `body["system"] = req.system` |
| #4 Per-request client | `gateway.py:983` | `async with httpx.AsyncClient(timeout=30) as client:` |
| #5 No catch-all tests | `tests/` | Zero matches for `proxy_passthrough` or `catch.all` |

---

## Repro Commands

```bash
# Verify preprocessing is skipped for OpenAI
cd /home/tony/projects/pichay
grep -n 'endpoint == "messages"' src/pichay/gateway.py

# Confirm no system handling in OpenAI adapter
grep -n 'system' src/pichay/providers/openai.py

# Confirm no catch-all tests
grep -rn 'proxy_passthrough\|catch.all' tests/

# Run existing tests to confirm baseline
uv run pytest tests/test_gateway_core.py tests/test_gateway_chaos.py tests/test_telemetry_contract.py -v
```

---

## Confidence

**High** for findings #1–#5. The code paths are unambiguous and the evidence is direct from source. **Medium** for #6 (tool format) — depends on whether the pipeline actually mutates tool definitions, which I did not trace fully.

---

## Follow-on Mission Recommendation

- **recommend_follow_on**: yes
- **expected_value**: medium
- **suggested_scope**: Trace the full OpenAI request lifecycle end-to-end — does any real OpenAI traffic flow through the gateway today? If so, audit SSE event parsing for OpenAI's streaming format (different event schema from Anthropic's `message_start`/`message_delta`). Also audit whether the telemetry contract tests (`test_telemetry_contract.py:test_request_metrics_contract_openai_stream`) use a real OpenAI payload shape or a synthetic one that happens to work despite the gaps.
- **stop_condition**: Once the OpenAI streaming event parser is confirmed correct (or confirmed unused in production), and the catch-all route has either been scoped to Anthropic-only or made provider-aware.
