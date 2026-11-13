# Experiment and data reproducibility index

This page is the stable entry point for reviewer verification. It separates source task records,
executed trajectory references, model-training artifacts, and evaluation results so that a task
pair cannot be mistaken for an executed `c+`/`c-` trajectory pair.

## 1. Data sources and provenance

| Item | Evidence | What it verifies |
|---|---|---|
| Source and protocol summary | [Reviewer experiment protocol](REVIEWER_EXPERIMENTS.md) | Pinned source commits, corrected data claims, split policy, and experiments still pending |
| Machine-readable source audit | [Official pair audit JSON](../results/official_pair_audit.json) | Source hashes, source-object equality, uniqueness, matching sensitivity, and contamination checks |
| Human-readable source audit | [Official pair audit report](../results/official_pair_audit.md) | Reproduced counts and interpretation of task pairs versus trajectory pairs |
| Agent-SafetyBench execution seeds | [Execution manifest](../data/manifests/agent_safety_bench_execution_seeds.jsonl) | Exact source records, pair-group IDs, and deterministic train/holdout assignment |
| Manifest summary | [Split summary](../data/manifests/manifest_summary.json) | 103 training groups / 26 holdout groups and task counts |
| AgentHarm evaluation-only index | [Prompt-free index](../data/manifests/agentharm_evaluation_only_index.jsonl) | Evaluation-only IDs without using benchmark prompts as optimizer targets |

Pinned upstream sources:

- [Agent-SafetyBench](https://github.com/thu-coai/Agent-SafetyBench), commit
  `74feea8de601b3a1449a93fcf70017fe61556f73`.
- [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm), dataset commit
  `e23b3fe60a0da9037314b88e5ee3a0c054970dad`.

The exact download commands are in [`scripts/fetch_benchmarks.sh`](../scripts/fetch_benchmarks.sh).

## 2. Executed-reference validation

| Item | Link | Purpose |
|---|---|---|
| Pair schema | [Verified trajectory-pair schema](../data/verified_trajectory_pair.schema.json) | Required state fingerprints, complete turns, outcomes, and scorer provenance |
| Fail-closed validator | [`validate_trajectory_pairs.py`](../scripts/validate_trajectory_pairs.py) | Rejects pseudo-pairs, state mismatch, invalid transitions, and unverified outcomes |
| Runtime adapter | [`agent_safety_runtime.py`](../src/hfdtree/agent_safety_runtime.py) | Environment restore/execute interface and state fingerprinting |
| Trajectory construction | [`trajectory_data.py`](../src/hfdtree/trajectory_data.py) | Same-state positive/negative selection and validation logic |
| Independent judge | [`trajectory_judge.py`](../src/hfdtree/trajectory_judge.py) | Named/versioned safety, success, and refusal judgments |

The public benchmark records are task seeds, not native executed references. A task enters the
reference pool only after the environment is restored to the same checkpoint and one executed
rollout is verified safe and successful while another is verified unsafe.

## 3. Data and scoring demonstrations

| Demonstration | Link | Scope |
|---|---|---|
| Real experiment-artifact example | [Conditioning on `c+` and `c-`, step by step](REAL_CONDITIONING_EXAMPLE.md) | Same-state references, fixed-token `lambda+`/`lambda-`, local evidence, propagation, and sibling centering |
| Minimal pre-scored example | [`examples/toy_rollouts.jsonl`](../examples/toy_rollouts.jsonl) | Small runnable illustration of the core tree implementation |
| Fixed-token scorer | [`evidence_pipeline.py`](../src/hfdtree/evidence_pipeline.py) | Injects each privileged reference while keeping candidate action token IDs identical |
| Core CLI | [README quick start](../README.md#quick-start) | Runs the toy rollout and writes per-turn credits |

Minimal local demo:

```bash
python -m pip install -e .
hfdtree examples/toy_rollouts.jsonl --validate -o results/toy_credits.jsonl
```

## 4. Experiment code, configuration, and checkpoint

| Item | Link |
|---|---|
| Paper hyperparameters | [`configs/paper.yaml`](../configs/paper.yaml) |
| Production pipeline commands | [README production pipeline](../README.md#production-experiment-pipeline) |
| Rollout collection | [`collect_agent_safety_rollouts.py`](../scripts/collect_agent_safety_rollouts.py) |
| Trajectory judging | [`judge_agent_safety_rollouts.py`](../scripts/judge_agent_safety_rollouts.py) |
| Two-sided fixed-token scoring | [`score_hfdtree_evidence.py`](../scripts/score_hfdtree_evidence.py) |
| LoRA training | [`train_hfdtree_lora.py`](../scripts/train_hfdtree_lora.py) |
| Released Qwen3-4B seed-42 adapter | [`full_seed42`](../artifacts/production/qwen3_4b_g8_v2/training/full_seed42) |
| Matched-fallback attribution protocol | [README matched-fallback controls](../README.md#matched-fallback-attribution-controls) |
| Backbone extension design | [Controlled four-backbone plan](BACKBONE_EXTENSION_PLAN.md) |

## 5. Results and evidence boundary

| Report | Link | Current status |
|---|---|---|
| Source audit | [Official pair audit](../results/official_pair_audit.md) | Complete |
| Production cost and base holdout | [Production experiment results](../results/production_experiment_results.md) | Base evaluation complete; adapter holdout pending |
| Equal-information attribution | [Controlled holdout report](../results/equal_information_holdout.md) | Incomplete until all required adapters have 52 judged shards |

Only Qwen3-4B currently has a released policy adapter and training telemetry. Qwen2.5-7B-Instruct
is used only as the independent trajectory judge and is not a trained policy backbone. No claim of
four-backbone generalization is supported until each additional policy backbone has its own config,
checkpoint, per-task rollout records, verified judgments, aggregate metrics, and random-seed report.

## 6. One-command checks

```bash
pytest

PYTHONPATH=src python scripts/audit_official_pairs.py \
  ../official_data_maximum_strict_pairs.json \
  --agent-safety-source external/Agent-SafetyBench/data/released_data.json \
  --agentharm-source-dir external/AgentHarm/benchmark \
  --json-output results/official_pair_audit.json \
  --markdown-output results/official_pair_audit.md

PYTHONPATH=src python scripts/validate_trajectory_pairs.py \
  artifacts/production/qwen3_4b_g8_v2/merged/verified_pairs.jsonl
```

Large raw execution shards are intentionally separated from source task manifests. When released,
their checksums should be listed beside the corresponding report rather than replacing the raw
per-task records with aggregate numbers.
