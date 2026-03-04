# Pichay Non-Inferiority Experimental Design

## Hypothesis

**Null hypothesis**: Baseline (no proxy) is superior to Pichay (managed eviction).

We attempt to reject this. Non-inferiority testing: we don't need to prove
Pichay is better, only that it is not meaningfully worse.

## Conditions

|                        | Baseline | Pichay |
|------------------------|----------|--------|
| **Normal** (in context)  | 2 runs   | 2 runs |
| **Long** (past compaction) | 2 runs   | 2 runs |

Both conditions must face the context wall. Without baseline-past-compaction,
we compare Pichay's worst case to baseline's best case, which proves nothing.

8 runs minimum per scenario. 3 scenarios for initial evaluation = 24 runs.

## Graduated Complexity Levels

Same spec, same blind tests, scaled pressure. Stop when both break.
The divergence point is the finding, not the ceiling.

**Level 1**: Task fits entirely in context. No eviction, no compaction.
Both should produce identical results. If Pichay is worse here (proxy
overhead, phantom tool interference), stop and fix before proceeding.

**Level 2**: Task fills context but doesn't overflow. Eviction/compaction
active but working set fits. Non-inferiority should hold easily.

**Level 3**: Task pushes past context. Both must manage memory. Baseline
compacts, Pichay evicts. Comparing strategies on equal terms — both lose
something, question is whether they lose the same things.

**Level 4**: Task substantially exceeds context. Multiple rounds of
compaction/eviction. Accumulated losses compound differently under
different strategies. Divergence appears here if it exists.

## Task Design

### Structure
- A written **spec** describing what to build
- A set of **blind tests** the model never sees
- The model builds from the spec, tests verify against the spec
- Tests are objective: pass or fail, no judgment needed for correctness
- Complexity scaled by codebase size and spec length, not by changing the task

### Why Blind Tests
- Model can't optimize for tests it can't see
- Must actually understand and retain the spec
- Edge cases in the test suite catch exactly the details lossy compression drops
- The test suite is ground truth; the spec is context pressure

### Scenario Types (from evaluation matrix)
Seven task types covering developer workflow:
1. Plan
2. Implement
3. Code review
4. Fix bug
5. Add feature
6. Document
7. Identify missing tests

Initial 3 for first evaluation (spanning read-heavy, write-heavy, mixed):
- Document a module (read-heavy, synthesis-dependent)
- Add a feature (write-heavy, multi-file)
- Debug a flaky test (mixed, investigative, high dead-result rate)

Any scenario can be expanded to push past compaction by scaling codebase
size and task complexity.

## Evaluation

### Objective Metrics (from proxy logs)
- Token consumption (input, output, cache)
- Context growth curve over session
- Fault rate (evictions that were re-requested)
- Cost estimate
- Blind test pass rate

### Qualitative Evaluation (blinded LLM ensemble)
For dimensions tests don't cover:
- **Correctness judge**: Is the implementation right?
- **Completeness judge**: Does it cover the full spec?
- **Coherence judge**: Is the approach architecturally sound?

If judges agree both are equivalent: strong non-inferiority signal.
If judges disagree about *which is better*: noise, not signal = non-inferiority.

### Human Calibration
- Sample cases where ensemble judges disagree or treatment scores notably
  different from baseline
- Human resolves hard cases
- Keeps human effort tractable even at scale

## Non-Inferiority Margin

Must be defined before running, not after seeing results. Candidates:
- Blind test pass rate: Pichay within X% of baseline
- Task completion: binary (did the agent finish the task?)
- Token cost: Pichay within Y% of baseline cost (may be higher due to faults)

The margin defines "not meaningfully worse." If Pichay passes 95% of tests
where baseline passes 97%, is that non-inferior? The margin answers this.

## Empirical Basis (from corpus study)

Reference string analysis across 63 sessions, 9,911 tool results:
- 75.6% of tool results are never re-referenced (safely evictable)
- Median re-reference distance: 111 turns
- Current age_threshold=4 catches only 10.6% of re-references
- Read results dominate the working set (75% of all re-references)
- Grep/Glob/WebSearch are 92-100% dead (discovery tools, use once)
- 24% fault rate observed in live session — policy architecture problem,
  not a tuning problem

These findings inform the eviction policy improvements but also establish
the baseline characteristics the evaluation should expect to see.

## Key Constraint

Do not design benchmarks that favor Pichay. The graduated complexity
ensures we test easy cases first. If Pichay is worse on easy tasks,
that's a real finding. The hard tasks are where divergence might appear,
but we earn the right to test there by passing the easy ones first.

## Open Questions

- [ ] Target codebase for evaluation (must be unfamiliar to the model,
      or accept prior knowledge as constant across conditions)
- [ ] Concrete spec + blind test suite for first scenario
- [ ] Non-inferiority margin values
- [ ] Whether to test across multiple models or focus on Claude
- [ ] Arbiter DSL as a candidate spec (spec a small language, both
      conditions implement parsers, blind test suite verifies conformance)
