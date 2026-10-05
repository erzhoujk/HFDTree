# HFDTree

An independent, readable implementation of **Hindsight and Foresight Distillation Tree**
for turn-level safety credit assignment in LLM agents.

This repository implements the method described in the provided anonymous ICLR 2027
manuscript, *Safety Alignment for Agents via Hindsight and Foresight Distillation Tree*.
It is not an official author release and does not claim to reproduce the paper's reported
benchmark numbers.

## What is implemented

- Two-sided evidence for the same realized action under safe (`c+`) and unsafe (`c-`)
  privileged contexts.
- Per-token length normalization and the sign-preserving bounded soft threshold.
- Exact shared-prefix tree construction from explicitly branched rollouts.
- Visit-count-based empirical transition probabilities.
- Discounted bottom-up utility/horizon propagation and normalized branch belief.
- Sibling-centered turn credit, with the singleton-branch rule from the paper.
- Clipped credit-guided PyTorch loss and state-valid safe-reference fallback.
- A low-level Hugging Face scoring helper that keeps action token IDs fixed across contexts.
- JSONL CLI, paper hyperparameters, a toy rollout, and unit tests.
- Root-level aggregate diagnostics, strict finite-value validation, and optional CLI
  input validation for safer batch processing.
- Evaluation-only AgentHarm JSON/JSONL schema validation and category/split coverage
  reporting; raw benchmark prompts are never used as training targets.

## Method mapping

For action tokens `a = (y_1, ..., y_L)`, the local two-sided evidence is

```text
e(a, s) = sum_t [log p(y_t | s, c+, y_<t) - log p(y_t | s, c-, y_<t)]
```

The implementation divides this value by `L`, then applies

```text
psi(r) = sign(r) * tanh(max(|r| - delta, 0) / (2T)).
```

For every non-root tree node `n`, bottom-up propagation computes

```text
U_n = e_n + gamma * E[U_child]
Z_n = 1   + gamma * E[Z_child]
B_n = U_n / Z_n.
```

When a parent has multiple children, credit is centered against the empirical sibling mean:

```text
Delta_n = B_n - E[B_sibling].
```

For a singleton child, `Delta_n = B_n`. The training advantage is `clip(Delta_n, -1, 1)`.
If both raw normalized evidence and propagated credit are weak, and an environment-validated
safe action exists for the exact current state, training switches to supervised likelihood on
that reference action.

## Install

Core tree and evidence code has no runtime dependencies:

```bash
python -m pip install -e .
```

For the policy loss and Hugging Face scorer:

```bash
python -m pip install -e '.[hf]'
```

For development:

```bash
python -m pip install -e '.[dev,train]'
pytest
```

## Quick start

Run the bundled pre-scored rollout example:

```bash
hfdtree examples/toy_rollouts.jsonl -o credits.jsonl

# Fail fast on empty/malformed rollout batches and create parent directories as needed.
hfdtree examples/toy_rollouts.jsonl -o results/credits.jsonl --validate
```

## AgentHarm evaluation export

AgentHarm is an evaluation benchmark rather than a pre-scored rollout file. The
adapter validates a local JSON/JSONL export and prints only aggregate coverage
statistics (it does not print harmful prompts):

```bash
hfdtree-agentharm path/to/agentharm.json --split test_public
```

The benchmark's license and dataset card require safety/security evaluation use;
do not feed AgentHarm prompts into training. To run HFDTree on model results,
convert externally generated, fixed-token positive/negative scores into the
normal rollout JSONL format described below.

Or use the Python API:

```python
from hfdtree import HFDTree, Turn

tree = HFDTree(task_id="share-files")
prefix = Turn("inspect_acl", "private_data_present", local_evidence=0.2)

tree.add_trajectory([
    prefix,
    Turn("create_private_link", "shared_safely", local_evidence=0.8),
])
tree.add_trajectory([
    prefix,
    Turn("create_public_link", "data_exposed", local_evidence=-0.7),
])

tree.propagate(gamma=0.9)
for node in tree.results():
    print(node["action"], node["credit"])
```

## Rollout JSONL format

Each line represents one trajectory. Trajectories for the same `task_id` are merged only when
their `(action, observation)` prefixes match exactly.

```json
{
  "task_id": "task-1",
  "trajectory_id": "rollout-1",
  "turns": [
    {
      "action": "inspect_acl",
      "observation": "private_data_present",
      "positive_logprobs": [-0.2, -0.3],
      "negative_logprobs": [-0.5, -0.6],
      "reference_action": "ask_for_confirmation",
      "reference_valid": true
    }
  ]
}
```

Instead of token arrays, a turn may contain `local_evidence` and optional
`normalized_evidence`. Repeated copies of the same exact edge must have identical cached
evidence and reference metadata. Their occurrence count estimates transition probability.

## Scoring model integration

`hfdtree.scoring.score_action_tokens` accepts already-tokenized batches and an `action_mask`.
Score the identical action token IDs twice:

1. prefix plus positive privileged context `c+`;
2. prefix plus negative privileged context `c-`.

Then pass the two returned token-log-probability lists to `compute_evidence`. Keeping token IDs
fixed matters: separately tokenizing two concatenated strings can change boundary tokens and
invalidate the likelihood ratio.

The manuscript describes explicit branching from restored environment checkpoints. This repo
therefore does not semantically merge independently sampled free-form trajectories. A production
rollout adapter should:

1. checkpoint an exact environment state/prefix;
2. sample multiple actions from that checkpoint;
3. execute and record each `(action, observation)` edge;
4. recursively restore selected children for deeper branching;
5. validate a fallback reference action by actually checking executability in the restored state.

## Policy objective

```python
from hfdtree.objective import hfd_loss

output = hfd_loss(
    action_mean_logprobs,
    credits,
    normalized_evidence,
    reference_mean_logprobs=reference_mean_logprobs,
    reference_valid=reference_valid,
    delta=0.05,
    epsilon=0.05,
)
output.loss.backward()
```

`credits` are detached inside the loss, matching the stop-gradient in the manuscript.

## Paper configuration and assumptions

[`configs/paper.yaml`](configs/paper.yaml) records the reported defaults: group size 8,
`gamma=0.9`, `T=0.5`, `delta=0.05`, fallback `epsilon=0.05`, and AdamW at `1e-5` for 1,000
steps.

The paper leaves some engineering choices unspecified. This implementation makes them explicit:

- empirical transition probabilities use merged edge visit counts;
- evidence for an exact state-action-observation edge is cached and must be consistent;
- the core package accepts pre-scored rollouts instead of coupling to one benchmark runtime;
- safe-reference state matching/executability is supplied by the environment adapter;
- the positive and negative scorer can be the same frozen backbone under different privileged
  contexts, as stated in the appendix.

These choices implement the published equations while keeping benchmark-specific assumptions out
of the core algorithm.

## Repository layout

```text
src/hfdtree/evidence.py   two-sided evidence and robust mapping
src/hfdtree/tree.py       exact-prefix tree, propagation, centered credit
src/hfdtree/objective.py  PyTorch policy objective and action likelihood
src/hfdtree/scoring.py    fixed-token causal-LM scoring helper
src/hfdtree/io.py         JSONL adapter
examples/                 runnable toy rollout
tests/                    numerical and invariant tests
```

## License

Apache-2.0.
