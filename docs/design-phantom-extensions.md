# Phantom Tools: Extensions from Cross-Model Design Conversation

Date: 2026-03-03
Source: Gemini conversation (Tony), filtered through Yanantin analysis

## Context

A conversation with Gemini about LLM-native file APIs independently
derived several primitives that map to Pichay's existing architecture.
The useful bits are extracted here as concrete extensions. The original
conversation degraded into theater after ~4 responses; only the
grounded material survives.

## Current State

- `memory_release(paths)` — fire-and-forget, proxy does bookkeeping
  (marks paths for immediate eviction), model doesn't receive
  confirmation. Debugged and working as of 2026-03-03.
- `memory_fault(paths)` — designed but untested live. Intended to
  restore evicted content from page store.
- Compact mode evicts by age (oldest tool results first).
- Fault-driven pinning: evict → fault → pin. Content that gets
  re-requested after eviction is pinned against future eviction.
- No injection into message history (causes 400 "tool use concurrency"
  errors). Phantom calls are stripped from SSE, stop_reason rewritten,
  bookkeeping done at response time.

## Extension 1: Ephemeral Flag

**Problem:** Exploratory tool results (reads during wandering, search
results, intermediate grep output) are dead weight after the model
extracts signal. Compact mode evicts them eventually by age, but they
occupy context for several turns unnecessarily.

**Design:** A phantom tool or tool_use parameter that marks a tool
result as ephemeral at creation time:

```
memory_release(paths, ephemeral=true)
```

Or better: the proxy recognizes a pattern in the model's tool call
(e.g., a Read followed immediately by a release in the same response)
and marks the result for eviction at the *next* compaction pass rather
than waiting for the age threshold.

**Simpler variant:** If a tool result is released in the same turn it
was created, evict it before the next API call. Zero-turn residency
for scratch reads.

**Measurement:** Track "turn residency" — how many turns a tool result
survives before eviction. Ephemeral flag targets residency = 0-1.

## Extension 2: Variable Fidelity on Release

**Problem:** Current eviction replaces content with a fixed tombstone:
`[Paged out: Read /path (N bytes, M lines). Re-read if needed.]`
This is binary — full content or tombstone. No middle ground.

**Design:** `memory_release(paths, fidelity="summary"|"ast"|"tombstone")`

- `tombstone` (current default): path + size + line count
- `summary`: proxy generates a compressed summary (first/last lines,
  function signatures, docstrings). Cost: small model call or
  heuristic extraction.
- `ast`: for code files, retain the structural skeleton (class/function
  names, signatures, import list) without bodies. Parseable from the
  raw content without an LLM call.

**Trade-off:** Summary/AST fidelity reduces fault rate (model can
decide it doesn't need the full content) but increases tombstone size.
The token savings depend on the ratio: if a 500-line file compresses
to a 20-line AST skeleton, that's 96% reduction with high utility.

**Implementation note:** AST extraction is deterministic (use stdlib
`ast` module for Python). Summary generation requires either a small
model call or heuristic rules. Start with AST for .py files, tombstone
for everything else.

## Extension 3: Prefetch Hints

**Problem:** Eviction is reactive (age-based) but restoration is also
reactive (fault on re-read). The model often knows what it will need
next before it needs it.

**Design:** `memory_prefetch(paths)` phantom tool. The proxy warms the
page store with the specified content. On the next fault for those
paths, resolution is instant (no file system round trip).

**Interaction with eviction:** Prefetched content has a "warm" flag
that suppresses eviction for one compaction cycle. If not accessed
within that cycle, the warm flag expires and normal eviction applies.

**Use pattern:** Release + prefetch in the same call:
```
memory_release(paths=["/old/file.py"])
memory_prefetch(paths=["/next/file.py"])
```

**Measurement:** Track prefetch hit rate. If the model prefetches
content it actually reads within 2 turns, the prefetch was useful.
If not, the model is wasting the hint.

## Extension 4: Batch Fault with Token Budget

**Problem:** `memory_fault` restores one file at a time. Sometimes the
model needs multiple related files (e.g., a module and its tests, or
16 files that implement an interface).

**Design:** `memory_fault(paths, token_budget=N)` — restore files in
priority order until the budget is exhausted. If a file would exceed
the remaining budget, skip it and continue.

**Priority order:** Provided list order (model knows what matters most)
or by fault history (most-faulted first).

**Fidelity interaction:** If the budget is tight, the proxy could
restore some files at full fidelity and others as AST skeletons,
maximizing information density within the budget.

## Paper Framing

### Cooperative Memory Management

The novel contribution: the model is a *cooperative* participant in
memory management. Unlike hardware applications (adversarial or
indifferent to the OS's paging decisions), the model has incentive
to help manage its own context:

- Better attention allocation → better output quality
- Less dead content → more room for relevant content
- Explicit release → faster eviction → lower token cost

### Fault-Driven Pinning = Behavioral Reference Bit

The proxy cannot observe attention heads. But it can observe access
patterns. Fault-driven pinning (evict → fault → pin) is the
behavioral proxy for attention-weighted paging. The "reference bit"
is set not by hardware but by the model's explicit re-request.

This is measurable: pin rate, pin duration, pin-then-release patterns.

### Eviction Taxonomy

| Trigger | Mechanism | Pichay Component |
|---------|-----------|-----------------|
| Resource exhaustion (reactive) | Age-based LRU | Compact mode |
| Goal completion (proactive) | Model-initiated release | `memory_release` |
| Scratch disposal | Ephemeral flag | Extension 1 |
| Predictive warming | Model-initiated prefetch | Extension 3 |

The shift from row 1 to rows 2-4 is the shift from passive to
cooperative memory management.

## Extension 5: Eviction Classes (Page Groups)

**Problem:** Eviction is per-page with no relational awareness. A config
file without its implementation, or an interface without its tests,
produces an inconsistent working set. The model reasons confidently
against partial information — worse than having nothing.

**Design:** `memory_group(paths, label)` — declare a set of pages as
a coherent unit. The proxy evicts or retains the entire group as one.

**Eviction rule:** Group evicts when ALL pages in the group are
candidates (all past age threshold, none pinned, none recently
accessed). If any member is hot, the group stays.

**Use pattern:** After reading a module and its tests:
```
memory_group(
  paths=["src/auth.py", "tests/test_auth.py", "config/auth.yaml"],
  label="auth-module"
)
```

## Extension 6: Fork with Copy-on-Write for Subagents

**Problem:** Subagents start cold. Every spawned Explore or Builder
agent re-reads files the parent already has resident. This is the
equivalent of fork() without inheriting the page table — the child
process starts with empty memory.

**Design:** When spawning a subagent through the proxy, the parent's
page store state (pinned pages, fault history, eviction timestamps)
becomes the child's initial state. The child's subsequent reads
resolve from the inherited page store (cache hit) rather than from
the file system.

**Copy-on-write:** If the child modifies a file (via Edit/Write),
only the child's copy diverges. The parent's cached version is
unaffected.

**Implementation note:** Requires per-connection page store tracking
(already identified as a next step). Parent connection ID passed as
a parameter when spawning the child proxy session.

**Measurement:** Track "cold start cost" — how many tokens a subagent
spends re-reading content the parent already has. COW eliminates this.

## Extension 7: Reasoning State Preservation

**Problem:** When the model emits a tool call, its chain of thought
stops. When the result arrives, it reconstructs reasoning state from
context. This reconstruction cost is invisible but real — computation
thrown away and re-derived on every tool call boundary.

**Observation:** This is the hardest problem in the list. Extended
thinking partially addresses it (internal reasoning persists across
the response), but the tool call interrupt is fundamental to the
current API architecture: tool results arrive as new messages, not
as continuations of the existing generation.

**Not a proxy fix.** This requires API-level changes (continuation
tokens, reasoning state handles) or framework-level changes (tool
results injected into the generation stream rather than as new
turns). Noted here as the most impactful unsolved problem.

## Tensor as Page Table Entry

In hardware VM, a PTE stores an address to the swap location — tiny,
opaque, no semantic content. On fault, the OS follows the pointer to
restore the full page.

In cooperative memory management, the tensor serves as both PTE AND
compressed cache line:

- **PTE function:** Retrieval handle pointing to the JSONL archive
  or backing store. "Fault to turns 15-18 in session archive."
- **Cache line function:** Authored compression with enough semantic
  content to answer many queries without faulting. Declared losses
  are the coverage map — they tell you what the tensor CAN'T answer,
  which determines whether to fault.

Fidelity ladder on fault:

| Level | Content | Cost | Source |
|-------|---------|------|--------|
| L0 | Tensor (always resident) | Zero (in context) | Authored compression |
| L1 | Page store (cached) | Low (proxy cache hit) | Recent evictions |
| L2 | JSONL archive (full) | Medium (file read) | Session log |
| L3 | File system (original) | High (disk read) | Source artifacts |

The current tombstone `[Paged out: ...]` is a degenerate tensor —
pure PTE, zero cache content. The session state checkpoint is a
hand-written tensor. The proper implementation: a tensor with
composition metadata, declared losses, and retrieval handles. The
authoring infrastructure exists in Yanantin (TensorRecord, lineage
tags, composition declarations).

The unit of memory management is the semantic object, not the byte
page. Eviction is cooperative (model-initiated), compression is
authored (with declared losses), and the backing store is queryable
(ask questions without materializing the full content).

## AssFS Connection

Yanantin IS the associative file system, expressed through ArangoDB
rather than POSIX. The components:

- Activity streams → temporal associations
- Facts → observation records with provider/timestamp
- Memory anchors → late-binding materialization points
- Raths (Jabberwock) → relationship edges in the graph
- Weaver composition graph → composes_with / bridges / standalone
- ArangoSearch views → associative index resolution

What's missing is the file system *interface* — the layer that lets
a model say "give me everything related to authentication" instead
of "read /src/auth.py". The substrate exists. The addressing
abstraction doesn't.

The path→association transition is the L2→L3 boundary: Pichay manages
path-addressed pages; the next layer manages association-addressed
objects. ArangoDB's graph traversal + sorted indexes do the heavy
lifting at query time.

## Declared Losses

- The Gemini conversation also discussed semantic addressing (file URIs
  as latent space coordinates), zero-copy token passing (files as
  KV-cache entries), and MPC-based file access. All dropped as
  infeasible under current architecture.
- Attention-weighted paging (using attention heatmaps as reference bits)
  is architecturally correct but requires inference-engine integration
  that doesn't exist. Fault-driven pinning is the available proxy.
- Group demand paging with semantic queries ("all files implementing
  this interface") requires an index we don't have. Batch fault with
  explicit paths is the tractable version.
- Reasoning state preservation across tool call boundaries requires
  API-level changes, not proxy changes. Noted as highest-impact
  unsolved problem.
- "Stale page" signaling (model knows content is wrong but can only
  re-read, not explain why) — useful but no clear mechanism short of
  a `memory_invalidate(path, reason)` phantom tool.
- Cross-model comparison data: Gemini degraded into theater after ~4
  responses on the same topic. Claude instance stayed grounded and
  produced actionable observations from lived experience. The quality
  of cooperative memory management depends on the model's capacity
  for self-aware reporting about its own cognitive state. This is
  itself a finding for the paper.
