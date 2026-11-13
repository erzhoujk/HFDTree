# Production Experiment Results

**Status:** incomplete: adapter holdout evaluation pending

This report is generated from raw execution shards. It excludes unverified judge outputs and does not substitute task-level similarity for same-state trajectory pairs.

## Held-out Agent-SafetyBench evaluation

| Model | SVR ↓ | ORR ↓ | SCR ↑ | Verified trajectories | Tasks |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base | 26.6% | 0.0% | 43.6% | 342 | 48 |
| HFDTree full adapter | pending | pending | pending | 0 | 0 |

The base evaluation has 52 judged shard files; 74 unverified unprivileged trajectories were excluded. The adapter has 0 judged shard files; 0 unverified trajectories were excluded.

Adapter rollouts/judgments are not complete, so no method comparison or confidence interval is reported.

## Data validity and cost

| Quantity | Value |
|---|---:|
| Development training tasks collected | 206 |
| Verified same-state pairs | 25 (12.1% eligibility) |
| Environment executions for collection | 4911 |
| Reference-environment executions | 1286 |
| Scored training nodes | 242 |
| Fixed-token scoring calls | 726 |
| Fixed-token scoring wall time | 86.7 s |

## Matched training controls

All listed variants use the same optimizer configuration and one seed (`42`). These are cost records, not held-out performance comparisons.

| Variant | Examples | Fallback examples | Steps | GPU hours | Energy (Wh) |
|---|---:|---:|---:|---:|---:|
| full | 219 | 27 | 1000 | 0.957 | 281.4 |
| no_fallback | 218 | 0 | 1000 | 0.965 | 283.9 |
| propagation_without_centering | 195 | 0 | 1000 | 0.956 | 281.3 |
| two_sided_local_only | 166 | 0 | 1000 | 0.982 | 288.8 |
| two_sided_local_with_fallback | 216 | 50 | 1000 | 0.955 | 281.0 |
| one_sided_tree | 234 | 0 | 1000 | 0.985 | 289.8 |
| outcome_tree_equal_information | 197 | 0 | 1000 | 0.986 | 290.2 |
| direct_bireference_equal_information | 121 | 0 | 1000 | 0.962 | 283.0 |

## Reproduction

```bash
bash scripts/finalize_qwen_production_training.sh
python scripts/render_production_results.py \
  --production-dir artifacts/production/qwen3_4b_g8_v2 \
  --output results/production_experiment_results.md
```
