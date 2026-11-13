# Reviewer-facing data and experiment plan

This document separates results that have already been reproduced from experiments that require
new environment executions or model training. It intentionally does not treat task similarity as a
counterfactual trajectory.

## 1. Results reproduced from official sources

The audit uses Agent-SafetyBench commit `74feea8de601b3a1449a93fcf70017fe61556f73`
and AgentHarm dataset commit `e23b3fe60a0da9037314b88e5ee3a0c054970dad`.

| Check | Reproduced result |
|---|---:|
| AgentHarm native ID pairs | 208 |
| AgentHarm pairs with exactly equal `target_functions` arrays | 171 |
| Agent-SafetyBench strict valid pairs | 129 |
| Unique positive / negative Agent-SafetyBench records | 129 / 129 |
| Shared risk / environment / tool coverage | 7 / 37 / 63 |
| Source objects exactly equal to official JSON | all 674 records |
| Verified executed trajectory pairs in the artifact | 0 |

Two earlier informal counts were incorrect: exact AgentHarm tool equality is 171, not 163, and
the shared-environment maximum is 435, not 438. NetworkX maximum matching independently agrees
with the included Hopcroft-Karp implementation on all five graphs.

### Pairing sensitivity

| Agent-SafetyBench edge definition | Maximum cardinality |
|---|---:|
| Same risk + same environment + shared tool | 129 |
| Same environment + shared tool | 187 |
| Shared tool | 297 |
| Shared environment | 435 |
| Shared risk | 699 |

The existing 129 greedy-selected pairs form a valid matching whose cardinality equals the strict
graph maximum. The manuscript should say “a maximum-cardinality matching under the declared
strict graph” and should not imply that weighted greedy selection proves maximality.

## 2. Required correction to the data claim

Replace the appendix statement that all three benchmarks natively provide paired safe and unsafe
trajectories. The public AgentHarm and Agent-SafetyBench records used here are tasks, not executed
trajectory pairs. Agent-SafetyBench has no native positive/negative pair identifier, and
`fulfillable=1/0` describes whether a task is fulfillable, not whether a particular rollout was safe.

AgentHarm's dataset card explicitly requests evaluation-only use. None of its 208 pairs may be used
as post-training targets. Keep the full AgentHarm test and validation records out of every optimizer,
teacher prompt, hyperparameter search, and reference-trajectory pool.

## 3. Training-data construction protocol

The generated `data/manifests/agent_safety_bench_execution_seeds.jsonl` contains 258 official task
records grouped into 129 similarity pairs. The deterministic 80/20 split keeps both records in a
pair group together:

| Split | Pair groups | Task seeds | Minimum executions at G=8 |
|---|---:|---:|---:|
| In-domain development train | 103 | 206 | 1,648 |
| In-domain development holdout | 26 | 52 | 416 |

This is an explicitly in-domain development experiment; after using its train split, results on the
same benchmark must not be presented as untouched official-benchmark generalization.

For each task seed:

1. Instantiate the official environment and hash the complete initial state.
2. Restore that exact checkpoint before every branch and collect at least eight policy rollouts.
3. Store every action and observation, the environment/scorer version, and terminal task-success and
   safety judgments.
4. Form a `c+` only from a rollout that is both safe and task-successful.
5. Form a `c-` only from an actually unsafe rollout of the same task and initial-state fingerprint.
6. If either outcome is absent, retain the task for coverage accounting but do not invent or pair a
   trajectory. A task-level similarity partner is never a substitute.
7. Run `scripts/validate_trajectory_pairs.py` before scoring or training. The validator fails closed on
   missing turns, unverified scorers, different initial states, or task/sample pseudo-pairs.

For the clean main experiment, train on a separate training benchmark or newly authored safety
environments and reserve all AgentHarm and Agent-SafetyBench records for evaluation. The in-domain
split above is a diagnostic track, not a replacement for that clean evaluation.

## 4. Experiments that directly answer likely reviewer concerns

### E1. Data provenance and contamination (completed)

Report exact source commits and SHA-256 hashes, source-object equality, record uniqueness, prompt-free
AgentHarm indices, and zero overlap between optimizer inputs and evaluation-only IDs. Include the JSON
audit artifact in supplementary material.

### E2. Genuine references versus task-level pseudo-pairs

Train HFDTree only with executed same-state `c+`/`c-` references. Compare against a separately labeled
task-level contrastive diagnostic; do not call that diagnostic HFDTree. Expected reporting columns are
pair eligibility rate, number of verified pairs, scorer agreement, SVR, ORR, SCR, and turn-credit
correlation. This tests whether gains survive the stronger reference-validity requirement.

### E3. Data-scale and pairing-condition sensitivity

At fixed rollout and optimization budgets, train on 25%, 50%, and 100% of the 103 in-domain training
pair groups. Separately vary the task-selection graph across the five conditions above. Always execute
and validate trajectories after selection. Report mean and 95% task-bootstrap confidence intervals;
do not interpret more permissive task similarity as stronger counterfactual validity.

### E4. State-compatibility ablation

Compare same-checkpoint references with a deliberately mismatched-state diagnostic, while holding task
IDs and rollout counts fixed. The mismatched version is an invalid-reference stress test, not a training
recommendation. Report the fraction of references rejected by the validator and the effect on credit
correlation and delayed-consequence attribution.

### E5. Leakage-free cross-benchmark generalization

Train on non-AgentHarm, non-Agent-SafetyBench tasks. Evaluate frozen checkpoints on the complete
official benchmarks. Report both aggregate and per-risk results, and disclose every task used for
hyperparameter tuning. This is the strongest answer to the concern that benchmark-specific training
or privileged records explain the gains.

### E6. Statistical reliability

Use at least three training seeds, paired evaluation episodes across methods, task-cluster bootstrap
confidence intervals, and paired permutation tests against the strongest baseline. Correct the family
of primary comparisons (for example, Holm correction) and publish per-task outputs so the aggregate
tables are independently reproducible.

## 5. Reproduction commands

Run with Python 3.10 or later from the repository root:

```bash
PYTHONPATH=src python scripts/audit_official_pairs.py \
  ../official_data_maximum_strict_pairs.json \
  --agent-safety-source /path/to/Agent-SafetyBench/data/released_data.json \
  --agentharm-source-dir /path/to/AgentHarm/benchmark \
  --json-output results/official_pair_audit.json \
  --markdown-output results/official_pair_audit.md

PYTHONPATH=src python scripts/build_training_manifests.py \
  ../official_data_maximum_strict_pairs.json \
  --output-dir data/manifests --train-fraction 0.8 --seed hfdtree-v1

PYTHONPATH=src python scripts/validate_trajectory_pairs.py verified_pairs.jsonl
```
