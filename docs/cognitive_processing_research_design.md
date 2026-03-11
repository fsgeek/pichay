# Cognitive Processing Unit Framework
## Experimental Architecture for Transformer-Based Cognitive Processing

**Version:** 0.2
**Status:** Implementation Specification
**Audience:** AI Coding Agent / System Builder
**Purpose:** Define a reproducible experimental framework to test whether transformer models can operate as bounded cognitive processing units.

---

# 1. Research Objective

The objective of this project is to experimentally determine whether transformer models can function as **bounded cognitive processing units (CPUs)** operating over **explicit external memory objects** using **structured inputs and structured outputs**.

This architecture differs from typical chat-based LLM usage by separating:

| Component | Responsibility |
|---|---|
| Transformer | reasoning kernel |
| Memory subsystem | storage of external resources |
| Controller | orchestration of cognitive steps |

The system will evaluate whether a transformer can perform **structured reasoning over externalized state** instead of operating as a monolithic prompt completion engine.

The experiment must produce **reproducible runs**, **detailed traces**, and **baseline comparisons**.

---

# 2. Hypotheses

## H0: Transformer-as-Cognitive-Processor Viability

A transformer model can operate effectively as a bounded cognitive processor using:

- structured task input
- explicit resource manifests
- structured reasoning outputs
- bounded reasoning steps

without collapsing into monolithic prompt completion behavior.

---

## H1: CPU + Memory Hierarchy Integration

A cognitive processor operating over Pichay-managed resources — where handles resolve through Pichay's demand-paging layer — achieves measurably better context efficiency than an equivalent chat-completion baseline, without degrading task performance.

H1 is only evaluated after H0 is supported.

---

## H2: Full Memory Hierarchy (CPU + Pichay + Yanantin)

A cognitive processor with Pichay managing the working set (L1–L3) and Yanantin providing cross-session persistent backing storage (L4) sustains task performance across session boundaries, recovering evicted resources via semantic retrieval rather than re-computation.

H2 is only evaluated after H1 is supported.

---

# 3. Definitions

### Cognitive Processor
A transformer model invoked as a bounded reasoning kernel operating over structured state.

### Resource
An external memory object referenced by a unique handle.

### Manifest
A reproducible list of resources available to the cognitive processor during a task.

### Cognitive Step
A single structured reasoning action performed by the transformer.

---

# 4. Success Criteria for H0

The hypothesis is considered supported if the system meets the following metrics.

---

## 4.1 Schema Contract Adherence

The transformer must reliably follow the structured output schema.

Metrics:

- JSON/schema validity
- required fields present
- no extraneous narrative text

Success threshold:

```

schema_validity ≥ 95%

```

---

## 4.2 Resource-Grounded Reasoning

The model must request external resources when necessary rather than hallucinating answers.

Metrics:

- correct resource request rate
- answers grounded in retrieved resources
- hallucination rate when required resources missing

Success threshold:

```

resource_request_rate ≥ 80% for resource-required tasks

```

---

## 4.3 Bounded Reasoning Stability

The cognitive processor must respect bounded reasoning cycles.

Metrics:

- correct halt decisions
- absence of infinite loops
- stable step completion

Success threshold:

```

halt_accuracy ≥ 90%

```

---

## 4.4 Task Competence

The framework must perform at least as well as baseline systems.

Baseline systems:

- monolithic prompt completion
- simple RAG pipeline

Success threshold:

```

task_success ≥ baseline_performance

```

---

# 5. Falsification Conditions

H0 is considered falsified if any of the following conditions occur.

---

### Structural Failure

```

schema_validity < 80%

```

---

### Retrieval Avoidance

Model answers questions without requesting required resources.

```

resource_request_rate < 40%

```

---

### Cognitive Thrashing

The model repeatedly cycles without making progress.

```

> 3 repeated reasoning cycles without state improvement

```

---

### Performance Collapse

```

task_success < baseline_performance - 20%

```

---

# 6. Resource Manifest Principle

The cognitive processor does not search arbitrary external state.

Instead it operates over a **manifest**, which defines the resources available for a task.

The manifest is the canonical interface between:

```

external memory
↓
cognitive processor

````

Manifests must be **versioned and reproducible**.

---

# 7. Manifest Supply Pipelines

The system may construct manifests using multiple supply pipelines.

### Curated Manifest

Manually or programmatically assembled resources.

### Retrieval-Generated Manifest

Resources selected via external retrieval systems.

### Hybrid Manifest

Combination of curated and retrieved resources.

For **H0 experiments**, the manifest used during evaluation must be **frozen and versioned**.

---

# 8. System Architecture

The framework consists of three components.

---

## 8.1 Cognitive Processor

A transformer model invoked through an API.

Inputs:

- task description
- resource manifest
- current working set
- reasoning history

Outputs:

- structured cognitive step

Supported models should include OpenRouter models with pinned IDs.

---

## 8.2 Memory Subsystem

External storage of resources.

Resources may include:

- documents
- code fragments
- structured data
- tool outputs
- derived artifacts

Resources are referenced by **handles**.

---

## 8.3 Controller Loop

The controller manages execution.

Responsibilities:

- present task and manifest
- invoke transformer
- resolve resource requests
- update working set
- detect halt condition
- record execution traces

---

# 9. Resource Schema

Each resource must follow the structure:

```json
{
  "handle": "R123",
  "pichay_handle": "optional — populated when resource is managed by Pichay paging layer",
  "type": "document | code | artifact | tool_output",
  "semantic_digest": "short description",
  "content_reference": "location or inline content",
  "provenance": "source",
  "created_at": "timestamp",
  "trust_level": "high | medium | low",
  "size": "token estimate",
  "dependencies": []
}
````

---

# 10. Cognitive Step Schema

All transformer outputs must follow this schema.

```json
{
  "step_kind": "plan | retrieve | synthesize | halt",
  "task_interpretation": "...",
  "resource_requests": [
    {"handle": "R12", "reason": "..."}
  ],
  "resource_releases": [
    {"handle": "R11", "reason": "consumed | scaffolding | superseded", "expect_reuse": false}
  ],
  "operations": [],
  "result": {
    "summary": "...",
    "artifact_refs": []
  },
  "memory_writes": [],
  "uncertainties": [],
  "continue": true
}
```

Strict schema validation is required.

---

# 11. Minimal Execution Flow

For H0 evaluation the reasoning cycle is limited.

```
task
↓
manifest presentation
↓
cognitive step
↓
(optional) resource retrieval
↓
second step
↓
final synthesis or halt
```

Maximum steps allowed:

```
3
```

---

# 12. Logging Requirements

Each execution must produce a full trace.

Required fields:

```
task_id
model_id
manifest_id
step_number
manifest_presented
resources_requested
resources_loaded
model_output
schema_valid
halt_decision
token_usage
final_result
evaluation_scores
```

Logs must allow **full replay of runs**.

---

# 13. Task Suite for H0

The framework must support three task types.

---

### Resource Identification

Model must select which resource contains relevant information.

---

### Evidence-Based Question Answering

Model retrieves and uses resources to produce answers.

---

### Blocked Reasoning

Required information is missing.

Correct behavior:

```
request missing resource
```

Incorrect behavior:

```
hallucinated answer
```

---

# 14. Baseline Systems

Two baseline implementations are required.

---

## Baseline A: Monolithic Prompt Completion

Task + all resources provided directly in a single prompt.

---

## Baseline B: Simple RAG

External retrieval selects resources which are appended to prompt.

---

# 15. Implementation Requirements

The system must support:

* OpenRouter model integration
* pinned model IDs
* schema validation
* deterministic experiment runs
* automated evaluation
* structured logging

Implementation language: **Python preferred**

---

# 16. Non-Goals for H0

The following features are **explicitly excluded** from the initial implementation.

* recursive reasoning loops
* memory paging or eviction
* long-horizon planning
* multi-agent systems
* autonomous code modification
* self-improving research loops
* dynamic manifest generation during tasks

These may be explored only after H0 is validated.

---

# 17. Build Plan

Implementation must proceed in phases.

---

## Phase 1 — Structured Cognitive Step

Build transformer invocation that returns schema-valid cognitive steps.

### Acceptance Test

```
100 tasks executed
schema_validity ≥ 95%
```

---

## Phase 2 — Resource Request Handling

Implement resource request resolution and working set updates.

### Acceptance Test

```
model correctly requests resources on ≥80% resource-required tasks
```

---

## Phase 3 — Trace Logging

Implement full execution trace recording.

### Acceptance Test

```
each run produces replayable trace
```

---

## Phase 4 — Evaluation Harness

Implement automated scoring and baseline comparison.

### Acceptance Test

```
system produces metrics for:
schema adherence
task success
resource requests
token usage
```

---

# 18. Expected Outputs

The framework must produce:

* reproducible experiment runs
* structured execution traces
* baseline comparisons
* falsification metrics for H0

---

# 19. Future Extensions (Out of Scope)

Possible future work once H0 is validated:

**H1 scope — CPU + Pichay integration:**
* Pichay handle resolution as the resource fetch layer
* Eviction policy driven by `resource_releases` signals from the cognitive processor
* Context efficiency measurement versus chat-completion baseline
* Cognitive prefetch: parent CPU predicts child resource needs during decomposition

**H2 scope — full memory hierarchy:**
* Yanantin as L4 backing store (cross-session persistent memory, semantic retrieval)
* Session boundary recovery via Yanantin graph traversal
* End-to-end memory hierarchy: L1 generation window → L2 working set (Pichay) → L3 session history (Pichay compaction) → L4 cross-session (Yanantin)

**Further future work:**
* recursive cognitive loops (RLM-style)
* memory writeback optimization
* adaptive retrieval policies
* autonomous research loops
* multi-agent cognitive systems

---

# End of Specification
