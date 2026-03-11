# Cognitive Transactions Synthesis (Phase 2/3/4)

Generated from fixed aggregate artifacts (no manual transcription).

## Artifact Inputs

- Phase 2 aggregate: `experiments/cognitive_transactions/phase2/phase2_20260311T150621Z/phase2_aggregate.json`
- Phase 3 aggregate: `experiments/cognitive_transactions/phase3/phase3_20260311T163854Z/phase3_aggregate.json`
- Phase 3 follow-up aggregate: `experiments/cognitive_transactions/phase3/phase3_synthetic_followup_20260311T165235Z/phase3_synthetic_followup_aggregate.json`
- Phase 3 tweak aggregate: `experiments/cognitive_transactions/phase3/phase3_synthetic_followup_tweak_20260311T170427Z/phase3_synthetic_followup_tweak_aggregate.json`
- Phase 4 aggregate: `experiments/cognitive_transactions/phase4/phase4_20260311T181454Z/phase4_aggregate.json`

## Core Claims (Bounded)

1. H0 viability is supported under induced protocol conditions: 3/4 Phase 2 models reached 100% validity with synthetic guidance, while 3/4 had 0% validity without guidance.
2. Guidance is load-bearing for OLMo-3: Phase 3 synthetic validity=86.7% vs synthetic_reduced=6.7%.
3. Contract/prompt tightening resolves legality drift in the tested synthetic protocol: follow-up validity=92.0% (23/25) and tweak retest validity=100.0% (50/50), with zero required-op misses and zero semantic rejections in the tweak retest.
4. Phase 4 realism confirms task-dependent token-budget effects (no truncations/content-filters in corrected run): tighter budget helps some tasks but hurts failure recovery.

## Claim Boundaries

- These results establish capability under this protocol family, not universal zero-shot cognitive transaction behavior.
- Multitask contradiction reconciliation remains the current boundary (validity below durability and recovery tasks in both token profiles).
- Budget effects are empirical and task-specific; there is no monotonic 'more tokens is always better' finding.
- Phase 4 conclusions are from the corrected run at `2026-03-11T17:30:18Z` and should be treated as single-run estimates pending confirmatory rerun.

## Phase 4 Budget Deltas (2000 minus 4000)

| Task | Valid @2000 | Valid @4000 | Delta |
|---|---:|---:|---:|
| failure_recovery | 73.3% | 80.0% | -6.7% |
| long_horizon_durability | 83.3% | 83.3% | 0.0% |
| multitask_benchmark | 66.7% | 72.2% | -5.6% |

## Repro Metadata (Phase 4 manifest)

- Git commit: `890d04af56cc700d656919d28517dda4c2f7c586`
- Git branch: `exp4`
- Git dirty: `True`

## Generated Tables

- `phase2_table.csv`
- `phase3_table.csv`
- `phase4_table.csv`
- `phase4_budget_delta.csv`
