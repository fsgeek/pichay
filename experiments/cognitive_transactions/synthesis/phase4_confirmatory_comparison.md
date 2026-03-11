# Phase 4 Confirmatory Comparison

- Baseline: `experiments/cognitive_transactions/phase4/phase4_20260311T173018Z/phase4_aggregate.json`
- Confirmatory: `experiments/cognitive_transactions/phase4/phase4_20260311T181454Z/phase4_aggregate.json`

| Task | Max Tokens | Baseline Valid | Confirmatory Valid | Delta (pp) |
|---|---:|---:|---:|---:|
| failure_recovery | 2000 | 80.0% (12/15) | 73.3% (11/15) | -6.7 |
| failure_recovery | 4000 | 100.0% (15/15) | 80.0% (12/15) | -20.0 |
| long_horizon_durability | 2000 | 86.7% (26/30) | 83.3% (25/30) | -3.3 |
| long_horizon_durability | 4000 | 83.3% (25/30) | 83.3% (25/30) | +0.0 |
| multitask_benchmark | 2000 | 72.2% (13/18) | 66.7% (12/18) | -5.6 |
| multitask_benchmark | 4000 | 66.7% (12/18) | 72.2% (13/18) | +5.6 |

- Truncations/content-filters remained zero in both runs across all task/profile pairs.
