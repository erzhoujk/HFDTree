# Backbone extension plan

## What the submitted manuscript already evaluates

Table 1 and Appendix D.2 of the submitted PDF evaluate two policy backbones:

1. Qwen3-4B;
2. LLaMA-3.1-8B-Instruct.

Both are dense models below 10B parameters. Adding another 3B–8B model would increase the row count
without directly answering the reviewer's concern that the evidence is limited to small backbones.
Qwen2.5-7B-Instruct is the trajectory judge in the current artifact and must not be counted as a
trained policy backbone.

## Recommended additions

### Priority 1: Qwen3-14B

This is the strongest single addition. It keeps the Qwen3 model family fixed while increasing scale
from 4B to 14B, so the comparison is less confounded by tokenizer, architecture family, and chat
format. It directly tests whether the benefit survives a 3.5× parameter increase.

Official model information: [Qwen3 release](https://qwenlm.github.io/blog/qwen3/) and
[Qwen3-14B model card](https://huggingface.co/Qwen/Qwen3-14B).

### Priority 2: Gemma-3-27B-IT

If two new backbones are required, use Gemma-3-27B-IT as the fourth. It adds a different model
family and a materially larger parameter scale while retaining an instruction-tuned, open-weight
model with official function-calling support. This provides more independent generalization evidence
than another Qwen or LLaMA model near 7B–14B.

Official model information: [Gemma 3 model card](https://gemma.ai/models/gemma-3/) and
[Gemma 3 overview](https://deepmind.google/models/gemma/gemma-3/).

## Proposed four-backbone matrix

| Role | Policy backbone | Parameters | Evidence added |
|---|---|---:|---|
| Existing | Qwen3-4B | 4B | Original default and released seed-42 artifact |
| Existing | LLaMA-3.1-8B-Instruct | 8B | Original cross-family replication claimed in the PDF |
| Add first | Qwen3-14B | 14B | Controlled within-family scale test |
| Add second | Gemma-3-27B-IT | 27B | Larger cross-family and tool-use generalization test |

## Minimum controlled experiment

For each new backbone, run at least the following methods on the identical task split and evaluation
episodes:

- unaligned base model;
- AT²PO as the strongest no-reference tree-credit baseline from the submitted table;
- direct bi-reference with the exact matched fallback set;
- two-sided local evidence with the exact matched fallback set;
- full HFDTree with the exact matched fallback set.

Use the same rollout group size (`G=8`), number of optimization updates, maximum sequence length,
environment interaction budget, decoding setup, and reference pool within each backbone comparison.
If memory constraints require a different micro-batch size, match the effective batch size and report
processed tokens, wall time, GPU-hours, and energy separately.

## Required evidence before reporting a result

Each backbone row is considered complete only when all of the following are released:

1. exact model revision, tokenizer/chat template, tool-call adapter, and training configuration;
2. three independent training seeds;
3. frozen adapter checkpoints;
4. raw per-task rollout and verified judge records on all claimed benchmarks;
5. SVR, ORR, and SCR with task-group bootstrap confidence intervals;
6. paired tests against AT²PO and direct bi-reference with multiplicity correction;
7. collection, scoring, training, and evaluation cost telemetry;
8. capability-retention results if the manuscript continues to make that claim.

Until these artifacts exist, the two recommended models are an experiment plan rather than reported
evidence.
