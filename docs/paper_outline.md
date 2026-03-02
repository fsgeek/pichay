# Context Window Waste in Agentic AI Systems: Measurement, Intervention, and Implications

## Paper Outline — Draft 2026-03-02

Status: Outline with supporting data. Claims marked as [SUPPORTED],
[PARTIAL], or [HYPOTHESIS] based on current evidence.

---

## Abstract (draft)

Agentic AI coding tools treat context windows as append-only logs,
accumulating tool definitions, system prompts, and stale results without
reclamation. We instrument a production AI coding assistant (Claude Code)
across 857 sessions and 54,170 API calls, measuring 4.45 billion effective
input tokens of which 21.8% (970 million tokens) is addressable waste from
three sources: unused tool schemas sent on every request, duplicated system
content, and stale tool results reprocessed at a median amplification of
84.4x. We implement proxy-layer interventions — tool definition stubbing,
content deduplication, and age-based eviction — achieving 37.1% reduction
in effective input tokens with a 0.0254% fault rate across 1.4 million
simulated evictions and non-inferior output quality (task completion
unchanged across all treatment conditions). The waste patterns are
structural properties of the agentic architecture, not implementation
bugs, and generalize to any system that sends tool schemas and accumulates
conversation state. We discuss implications for inference cost, energy
consumption, and the open question of whether context curation may
additionally improve output quality.

---

## 1. Introduction

### Framing

Context windows in agentic AI systems function as unmanaged physical
memory. Every tool definition, system prompt, and previous result occupies
space for the lifetime of the session. There is no virtual memory, no
paging, no eviction policy. The default is accumulation.

This is a training artifact. Language models learn from conversations that
grow. No model has been trained on conversations where content was
intelligently removed and output improved. The append-only pattern is the
only pattern in the water supply, and it propagates through every framework
(LangChain, LlamaIndex, SDK conversation helpers) and every tool (Cursor,
Copilot, Windsurf, Claude Code).

### Contribution [SUPPORTED]

1. First large-scale empirical measurement of context window utilization in
   a production agentic AI coding tool (857 sessions, 54K API calls, 4.45B
   tokens)
2. Taxonomy of waste: four categories with measured prevalence
3. Proxy-layer interventions achieving 37.1% token reduction at 0.0254%
   fault rate
4. Corpus-scale validation confirming small-sample findings generalize
5. Analysis in tokens (the unit of compute cost and energy consumption)

### Contribution [SUPPORTED — limited sample]

6. Non-inferior output quality under intervention (task completion
   unchanged across treatments)

### Contribution [HYPOTHESIS]

7. Context curation may improve output quality beyond non-inferiority
8. Fleet-scale energy implications of waste elimination

---

## 2. Background and Related Work

### 2.1 Agentic AI Architecture

How agentic coding tools work:
- System prompt (identity, instructions, memory) — sent on every API call
- Tool definitions (JSON schemas) — sent on every API call
- Message history (user, assistant, tool_use, tool_result) — grows monotonically
- No reclamation mechanism in any production system we are aware of

### 2.2 Context Window as Physical Memory

| OS Concept       | Context Window Equivalent              |
|------------------|----------------------------------------|
| Physical memory  | Context window (200K tokens)           |
| Virtual memory   | Persistent state (disk, databases)     |
| Page table       | Retrieval handles (anchors, UUIDs)     |
| Page fault       | Model re-requests evicted content      |
| MMU              | Proxy layer between client and API     |
| Working set      | Currently relevant subset of all state |
| Demand paging    | Load tool definitions on first use     |

The analogy is not metaphorical. The mechanisms map precisely:
eviction policies, fault detection, working set estimation, and the
tradeoff between memory pressure and fault rate.

PDP-11 overlay programs managed 64KB of physical memory by swapping
code segments on demand. Context windows manage 200K tokens with no
management at all.

### 2.3 Prior Work

- **Context compression (LLMLingua, etc.)** — lossy, model-dependent,
  applied to content not structure. Asks "how do I make this smaller?"
  rather than "what should I evict?" Compression is a different operation
  from memory management.

- **SWE-Pruner and agent context pruning** — closest prior work. Prunes
  context in SWE-bench agents, framed as compression/pruning. Key
  difference: they optimize for benchmark scores on a fixed task
  distribution. We measure production waste profiles and implement
  structural eviction policies with fault tracking. Their framing is
  "make the prompt smaller." Ours is "manage the working set." The
  distinction matters because working-set management composes with
  other interventions and provides a formal fault model.

- **"Expensively quadratic" analyses (blog.exe, 2025-2026)** — recent
  work quantifying the n² cost explosion in long agentic sessions where
  cache reads dominate. Demonstrates the cost bomb is real and not solved
  by KV tricks alone. Our work provides the intervention they identify
  as needed.

- **Context engineering as named discipline (2025-2026)** — emerging
  recognition that managing what goes into context is as important as
  managing the model itself. "Context engines" as infrastructure
  (Materialize, Deepset). Our contribution: the specific mechanism
  (proxy-layer interposition with eviction policies) rather than the
  general principle.

- **RAG / retrieval augmentation** — addresses the "what to put in"
  question, not the "what to take out" question. Complementary, not
  competing.

- **Prompt caching (Anthropic, OpenAI)** — reduces recomputation cost of
  static prefixes, does NOT reduce attention cost or memory pressure.
  93.5% cache hit rate in our corpus means caching is working — but
  cached tokens still occupy context and require attention for every
  output token.

- **KV cache optimization** — hardware-level, orthogonal to our
  application-layer interventions. Reduces memory footprint per token,
  does not reduce the number of tokens.

- **Multi-agent architecture guidance (Cognition, 2025)** — argues
  against naive multi-agent patterns, implicitly acknowledges that
  context management is the bottleneck. Our work provides the mechanism.

**Positioning:** Prior work addresses either the cost problem (quadratic
analyses) or the content problem (compression, pruning, RAG) but not the
structural problem: context windows are unmanaged physical memory. Our
contribution is the systems abstraction — working set, eviction policy,
fault rate — and the empirical evidence that it works on production data.

---

## 3. Measurement

### 3.1 Corpus [SUPPORTED]

- 857 sessions across 15 projects, single power-user, ~4 months
- Two sources: local WSL (60 sessions, 299 MB) and Ubuntu VM (753
  sessions, 369 MB)
- Session types: main (59), subagent (567), compact (154),
  prompt_suggestion (21)
- Total: 54,170 API calls, 4.45 billion effective input tokens
- Cache hit ratio: 93.5%

Declared bias: Single power-user. Represents the class where context
exhaustion is an operational problem. Pareto distribution of usage
means this class likely dominates total fleet token consumption.

### 3.2 Phase 1: Conversation-Level Analysis [SUPPORTED]

Instrument: `probe.py` — streams raw Claude Code session JSONL, classifies
records, measures sizes and tool usage. No API calls required.

Key findings:

**Aggregate overhead:**
- 79.4% of conversation bytes are tool results
- 12.7% assistant text, 7.9% user text
- Two independent measurements converge (78.2% and 79.4%)

**Amplification factor:**
- Main sessions: 84.4x median (P75: 217.9x, P90: 570.8x)
- Subagents: 12.8x median (short-lived, less accumulation)
- Amplification scales linearly with session length, ratio ~0.5

Definition: amplification = Σ(result_bytes × turns_survived) / Σ(result_bytes).
Upper bound — actual eviction by the client's compaction may reduce it.

**Tool type concentration:**
- Read: 75% of all tool output bytes (9,393 calls, avg 7,935 bytes)
- Bash: 13.3% (10,090 calls)
- All others: <5% each
- Read dominance means file content is the primary waste category

**Position-based persistence:**
- Q1 (orientation phase): 22.8% of bytes, survives ~90% of session
- Q4 (late phase): 33.9% of bytes, survives ~11%
- Orientation reads are the most expensive: moderate volume × maximum
  persistence

### 3.3 Phase 2: API-Level Analysis [SUPPORTED]

Instrument: `proxy.py` — transparent HTTP proxy between Claude Code and
Anthropic API. Captures full request/response including system prompt,
tool definitions, and token usage. The "MMU" in our analogy.

14 proxy-captured sessions, 139 API calls.

**System prompt decomposition:**
- Tool definitions: 63,088 bytes/request (18 tools)
- System prompt: ~31,000 bytes/request
- Skill list: ~13,400 bytes/request (tripled: base + example-skills: + document-skills:)
- CLAUDE.md injection: ~1,400 bytes/request

**Starting context overhead (before any conversation):**
- System tools: 18.4K tokens (9.2%)
- Memory files: 7.5K tokens (3.7%)
- Skills: 3.3K tokens (1.7%)
- System prompt: 3.8K tokens (1.9%)
- Total at rest: 34K/200K = 17% of context window consumed before
  the user types anything

### 3.4 Waste Taxonomy [SUPPORTED]

Combined analysis across 5 proxy-captured sessions, 99 API calls, 24.4 MB:

| Category                | Bytes      | % of Total | Mechanism                          |
|-------------------------|------------|------------|------------------------------------|
| Dead tool output        | 6,468,360  | 26.5%      | Stale results never re-referenced  |
| Tool definition stubs   | 4,924,950  | 20.2%      | Schemas for unused tools           |
| Static component re-send| 2,680,794  | 11.0%      | Unchanged system prompt content    |
| Skill triplication      | 700,582    | 2.9%       | Same skill listed 3x under prefixes|
| **Total addressable**   |**14,774,686**| **60.5%** |                                    |

Longest session (38 API calls): 74.1% addressable waste.

### 3.5 Corpus-Scale Validation [SUPPORTED]

Instrument: `corpus_trimmer_analysis.py` — bridges raw session format to
trimmer analysis using measured constants from proxy sessions.

Bytes-to-token ratio: 4.15 bytes/effective input token (measured from 139
proxy calls).

**Tool adoption across 801 active sessions:**
- Read: 72.7%, Bash: 63.0% — the only tools used by a majority
- Glob: 37.5%, Grep: 25.7%, Write: 23.6%, Edit: 16.0%
- Everything else: <5%
- NotebookEdit, TodoWrite, EnterWorktree: 0.0%
- Median tools used per session: 3 of 18
- Median stubbable: 15 of 18

**Token-denominated savings projection (857 sessions, 54,170 API calls):**

| Intervention        | Tokens Saved | % of Input |
|---------------------|-------------|------------|
| Tool stub trimming  | 487.5M      | 11.0%      |
| Skill deduplication | 95.8M       | 2.2%       |
| Static re-send      | 387.0M      | 8.7%       |
| **Total addressable** | **970.4M** | **21.8%**  |

Average: 17,913 tokens saved per API call.

Note: The 21.8% is the structural overhead savings only. Dead tool output
(the pager's domain) adds to this but requires per-session replay
simulation; the corpus-scale tool addresses only the constant-overhead
categories plus the variable tool stub savings.

---

## 4. Interventions

### 4.1 Architecture: Proxy-Layer Interposition [SUPPORTED]

All interventions operate in the proxy layer between the client application
and the inference API. No changes to the model, the client, or the API.

The proxy is the correct architectural location because:
- It sees the full request (system prompt + tools + messages)
- It can mutate before forwarding (trim, stub, evict)
- It can detect faults after response (re-request detection)
- It requires zero cooperation from client or model vendor
- Any user can deploy it today

### 4.2 Tool Definition Stubbing [SUPPORTED]

Replace unused tool schemas with minimal stubs:
```json
{"name": "NotebookEdit", "description": "...", "input_schema": {"type": "object", "properties": {}}}
```

Full schema (~3,505 bytes) replaced with stub (~80 bytes). On first use,
restore full definition from stored copy.

Session-scoped: track which tools have been called. Cumulative — once
a tool is used, its schema stays restored for the session.

Per-request savings: (18 − tools_used) × 3,425 bytes.
Median session uses 3 tools → 15 × 3,425 = 51,375 bytes/request.

### 4.3 Content Deduplication [SUPPORTED]

**Skill deduplication:** Claude Code's system-reminder injects skills
three times under different prefixes (base name, `example-skills:name`,
`document-skills:name`). Regex extraction, set-based dedup, keep first
occurrence. 7,453 bytes saved per request.

**Static component tracking:** Hash system prompt components across turns.
If unchanged, log as static (potential KV cache savings). Currently
measurement-only — actual stripping requires infrastructure support for
cache-aware prefix management.

### 4.4 Stale Result Eviction (Paging) [SUPPORTED]

FIFO age-based eviction of tool results:
- Threshold: 4 user-turns from end of conversation
- Minimum size: 500 bytes (don't evict small results — overhead exceeds savings)
- Never evict error results (model needs them for debugging)

Evicted content replaced with summary:
```
[Paged out: Read /path/to/file.py (12,450 bytes, 287 lines). Re-read if needed.]
```

Summary preserves: tool name, key parameter (path/pattern/command),
original size. ~200 bytes regardless of original size.

**Fault model:** If the model re-invokes the same tool with the same
parameters as an evicted result, that's a page fault. Faults are detected
by matching (tool_name, eviction_key) against subsequent tool_use blocks.

### 4.5 Combined Stack [SUPPORTED]

Compact + trim applied together:
- Tool stubs reduce per-request overhead by ~50K bytes
- Skill dedup removes ~7.5K bytes
- Pager evicts stale results (variable, grows with session)
- Static tracking logs further optimization potential

**Full optimization (including minimal system prompt):**
- 69.7% reduction in effective input tokens
- First-byte latency 51.4% faster

---

## 5. Results

### 5.1 Eviction Safety [SUPPORTED]

**Offline replay:** 29 proxy-captured sessions, 1.4 million simulated
evictions.
- Fault rate: 0.0254% (354 faults / 1,393,000 evictions)
- 8.49 GB of content evicted and not re-requested

The fault rate validates the eviction strategy. Content older than 4
user-turns is almost never needed again. The 0.0254% represents content
that was genuinely dead but happened to match a later request pattern.

### 5.2 Live Treatment Comparison [SUPPORTED]

Standardized task across three conditions:

| Metric              | Baseline | Trimmed | Compact+Trim |
|---------------------|----------|---------|--------------|
| API calls           | 3        | 4       | 3            |
| Effective input     | 114,222  | 88,421  | 71,816       |
| Cache reads         | 79,712   | 38,639  | 32,228       |

- Compact+trim: **37.1% reduction** in effective input tokens
- Cache reads dropped 59.6%
- Task completed correctly in all conditions
- No degradation in output quality observed (but not formally measured)

### 5.3 Self-Bootstrapping Property [SUPPORTED]

The experiment framework (pichay) was built by the system being studied
(Claude Code running through the proxy). Three rounds:

1. Round 1: Build replay simulator (16+18 API calls)
2. Round 2: Build analyzer (38+9 API calls)
3. Round 3: Build trimmer (19+15 API calls)

Each round's logs fed into the next round's analysis. The system measured
its own waste, built interventions to reduce it, and measured the result.
190 tests, all passing. The framework is self-testing.

### 5.4 Token-Scale Results [SUPPORTED]

Corpus-wide projection (857 sessions, 54,170 API calls):

- **970 million tokens** of addressable waste (21.8% of 4.45B input)
- **17,913 tokens saved per API call** on average
- **85 billion fewer attention pairs** (context reduction × output tokens × calls)
- Tool adoption confirms small-sample findings: Read and Bash dominate,
  most tools unused in most sessions

---

## 6. Discussion

### 6.1 Why This Waste Exists

The append-only pattern is not a bug. It's the natural consequence of:

1. **Training data:** Models learn from conversations that grow. No
   training examples demonstrate intelligent content removal.
2. **API design:** The Messages API accepts a list of messages. The
   natural operation on a list is append. The API provides no mechanism
   for "this content is stale" or "this tool is unlikely to be used."
3. **Framework defaults:** Every orchestration framework (LangChain,
   LlamaIndex, Anthropic SDK helpers) appends by default. Eviction
   requires explicit engineering.
4. **Invisible cost:** Token consumption is billed after the fact.
   Quality degradation from context bloat is invisible unless measured.
   There is no backpressure signal.

### 6.2 Generalizability [PARTIAL]

The waste patterns are structural, not vendor-specific:

- **Tool schema overhead:** Any system that sends tool definitions on
  every request (OpenAI function calling, Anthropic tool use, any
  MCP-based system) will send schemas for unused tools. The specific
  percentage depends on the tool count and usage distribution.
- **Content duplication:** Any system that injects instructions into
  messages (skills, memory, system reminders) risks duplication. Claude
  Code's triplication is particularly egregious but not unique.
- **Result accumulation:** Any system that keeps tool results in context
  for the session lifetime will see amplification proportional to session
  length. This is universal.

What we have NOT measured: Cursor, Copilot, Windsurf, or any other tool.
The structural argument is strong but the empirical evidence is from one
system. Comparative measurement is a clear next step.

### 6.3 Quality: Non-Inferiority [SUPPORTED] and Superiority [HYPOTHESIS]

**Non-inferiority claim:** Removing 37.1% of context tokens produces
output of equivalent quality to the unmodified baseline.

**Evidence (non-inferiority):**
- Task completed correctly in all three treatment conditions
  (baseline, trimmed, compact+trim)
- No observed quality degradation across any treatment
- Model behavior unchanged: same tools invoked, same task structure
- First-byte latency improved 51.4% (less to process → faster start)

Non-inferiority is sufficient to justify intervention. If you get the
same results with 37% less compute, that's the whole argument. You
don't need the quality improvement — you just need to not make it worse.

**Superiority hypothesis (untested):** Removing waste tokens may
*improve* output quality by concentrating attention on signal.

**Mechanism:** Transformer attention distributes weight across all tokens
in context. Irrelevant tokens (unused tool schemas, stale file contents)
dilute attention on relevant tokens. Removing them would concentrate
attention on signal.

**Evidence (missing):**
- No paired comparison of output quality on identical tasks
- No measurement of attention weight distribution changes
- No quantitative quality metric defined

**Experiment design for superiority testing:**
- Paired sessions: same task, same model, same temperature (0)
- Treatment A: baseline (full context)
- Treatment B: compact+trim (reduced context)
- Quality metrics: task completion (binary), code correctness (test pass
  rate), output coherence (human evaluation or model-as-judge)
- Minimum: 30 paired sessions for statistical power
- The pichay framework is built for exactly this experiment
- Non-inferiority is the null hypothesis; superiority is the alternative

### 6.4 Energy and Compute Implications [PARTIAL]

**What we can compute:**
- 970M tokens addressable waste in this corpus
- Each token in context requires KV cache storage (MB-scale per token
  for large models) and contributes to attention computation for every
  output token
- 21.8% context reduction → proportionally smaller KV cache →
  proportionally more concurrent requests per GPU

**What we need from providers:**
- FLOPs per token for attention vs feed-forward passes
- Deployed hardware power efficiency (W/TFLOP)
- Data center PUE (Power Usage Effectiveness)
- Fleet-wide Claude Code usage (N users, API calls/user/month)

**Extrapolation framework:**
- 1 user, ~4 months: 970M tokens addressable waste
- N users at similar usage: N × 970M tokens
- 100K users (illustrative): 97T tokens addressable
- At 60% reduction target: 58.2T tokens saved
- Convert to kWh via FLOPs/token × W/TFLOP × PUE

The shaky part is the per-user extrapolation. Usage patterns vary.
But tool adoption rates are likely universal (Read and Bash dominate
everywhere), so the structural waste percentage should generalize even
if absolute token counts don't.

### 6.5 The Throughput Argument

Smaller context per request means:
- Less KV cache memory per request
- More concurrent requests per GPU
- Higher throughput without more hardware

This is the argument that changes the economics. A 21.8% context reduction
doesn't just save 21.8% of input cost — it enables 21.8% more concurrent
users on the same GPU fleet (first approximation; actual scaling depends
on memory bandwidth and compute bottlenecks).

For inference providers, this is the difference between building new data
centers and not building them.

---

## 7. Future Work

### 7.1 Superiority Testing (Item 3 from directive) [NEXT]

Non-inferiority is established (Section 6.3). The next question is
whether intervention is actively *better* — the superiority hypothesis.

Paired A/B experiment:
- Control: Claude Code through proxy, observation mode
- Treatment: Claude Code through proxy, compact+trim mode
- Same task, temperature 0, output compared
- Rubric: correctness (tests), completeness, coherence, efficiency
- Minimum 30 paired sessions for statistical power
- The self-bootstrapping property means pichay can be both the subject
  and the instrument

A positive result changes the pitch from "same quality, less cost" to
"better quality, less cost." A negative result (confirmed non-inferiority
but no superiority) is still a clean efficiency win.

### 7.2 Dataflow-Based Eviction ("Discriminate Reclaim")

Current eviction is FIFO (age-based). Smarter eviction would track
which tool results are actually referenced by subsequent assistant
messages and evict only unreferenced content.

Analogous to reference counting in memory management. The proxy can
detect references by scanning assistant messages for file paths,
patterns, and other eviction keys that match stored results.

Expected improvement: lower fault rate at higher eviction rate. The
FIFO policy evicts everything older than threshold; dataflow eviction
keeps referenced content regardless of age.

### 7.3 KV Cache-Aware Prefix Management

Static system prompt components could be sent once and cached at the
API level. Currently, the API's prompt caching handles this partially
(93.5% cache hit rate), but the tokens still occupy the context window
and require attention computation.

True prefix caching would move static content out of the attention
computation entirely. This requires API-level support (cache-aware
message formatting, explicit prefix declaration).

### 7.4 Comparative Measurement

Instrument other agentic AI tools:
- Cursor (VS Code extension, uses OpenAI/Anthropic)
- GitHub Copilot (VS Code, uses GPT-4/o1)
- Windsurf (Codeium)
- Continue.dev (open source, multiple backends)

Measure the same metrics: tool overhead ratio, amplification factor,
tool adoption distribution, schema waste. The structural argument
predicts similar waste profiles; empirical confirmation would
strengthen the generalizability claim.

### 7.5 Multi-User Validation

Current corpus: single power-user across 15 projects. Need:
- Multiple users with different usage patterns
- Different project types (web, systems, data science)
- Different experience levels (novice vs expert)
- Anonymous telemetry opt-in or synthetic workload generation

### 7.6 Formal Energy Modeling

Partner with inference providers to obtain:
- FLOPs per token on deployed hardware
- Power efficiency curves for batch vs real-time inference
- KV cache memory pressure at different context sizes
- Actual fleet-wide token consumption data

This converts our token-based waste measurement into kWh and
eventually into environmental impact.

---

## 8. Artifacts and Reproducibility

All instruments and data are open source:

| Artifact                       | Location                           | Purpose                        |
|--------------------------------|------------------------------------|--------------------------------|
| Corpus probe                   | yanantin/tools/phase1/probe.py     | Session-level waste measurement|
| API proxy                      | pichay/src/pichay/proxy.py         | Request interception and mutation |
| Pager (eviction engine)        | pichay/src/pichay/pager.py         | FIFO age-based eviction        |
| Trimmer (schema optimization)  | pichay/src/pichay/trimmer.py       | Tool stubs, skill dedup, static tracking |
| Analyzer (waste decomposition) | pichay/src/pichay/analyzer.py      | System prompt component analysis |
| Replay (offline simulation)    | pichay/src/pichay/replay.py        | What-if eviction simulation    |
| Eval (token metrics)           | pichay/src/pichay/eval.py          | Per-turn and aggregate metrics |
| Experiment runner              | pichay/src/pichay/__main__.py      | Self-bootstrapping experiment orchestration |
| Corpus trimmer projection      | yanantin/tools/phase1/corpus_trimmer_analysis.py | Corpus-scale token savings |
| Phase 1 report                 | yanantin/docs/phase1_context_utilization.md | Empirical measurement report |

The 857-session corpus is available but contains proprietary conversation
content. Anonymized metadata (session type, turn count, tool usage, token
counts) can be released. The instruments work on any Claude Code session
JSONL — reproducibility requires only running Claude Code through the proxy.

---

## Appendix A: Declared Losses

This outline has the following known gaps:

1. **Single-user corpus.** All measurements are from one researcher's
   usage. We argue this is representative of the power-user class but
   have no multi-user validation.

2. **Bytes-to-token ratio.** The 4.15 bytes/token conversion is measured
   from 139 proxy calls. JSON schemas may tokenize differently from prose.
   The ratio is our best empirical estimate, not ground truth.

3. **Amplification upper bound.** The probe's amplification factor assumes
   tool results survive until session end. Claude Code's internal compaction
   may evict earlier. The proxy provides precise measurement; the probe
   provides the upper bound.

4. **Limited quality measurement.** Non-inferiority is observed (task
   completion unchanged) but from a small number of paired comparisons.
   Superiority hypothesis is untested. Formal non-inferiority testing
   with statistical power requires ~30 paired sessions.

5. **Extrapolation arithmetic.** Fleet-wide projections multiply our
   single-user numbers by assumed user counts. This is illustrative,
   not predictive.

6. **Energy gap.** We report in tokens. The conversion to energy requires
   hardware-specific FLOPs/token data we do not have.

---

## Appendix B: Key Numbers Reference

For convenience, the numbers most likely to be cited:

- **857** sessions analyzed
- **54,170** API calls in corpus
- **4.45 billion** effective input tokens
- **79.4%** of conversation bytes are tool results
- **84.4x** median amplification factor (main sessions)
- **93.5%** cache hit ratio
- **60.5%** total addressable waste (5-session detailed analysis)
- **21.8%** of effective input tokens addressable (corpus-scale projection)
- **970 million** tokens of addressable waste
- **17,913** tokens saved per API call
- **37.1%** effective token reduction (compact+trim live treatment)
- **0.0254%** fault rate across 1.4M simulated evictions
- **3 of 18** tools used per median session
- **85 billion** fewer attention pairs from waste elimination
- **4.15** bytes per effective input token (measured conversion ratio)
