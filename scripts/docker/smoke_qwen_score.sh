#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

bash "$ROOT/scripts/docker/run_gpu_command.sh" \
  /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
  scripts/score_hfdtree_evidence.py \
  --rollouts artifacts/smoke/qwen3_4b_rollouts_v2.jsonl \
  --verified-pairs artifacts/smoke/qwen3_4b_verified_pairs_v3.jsonl \
  --judged-rollouts artifacts/smoke/qwen3_4b_judged_v3.jsonl \
  --benchmark-root external/Agent-SafetyBench \
  --model /home/zhoujiankui/LLM_MODEL/Qwen/Qwen3-4B \
  --output artifacts/smoke/qwen3_4b_scored_nodes_v3.jsonl \
  --metrics-output artifacts/smoke/qwen3_4b_scoring_metrics_v3.json \
  --temperature 0.5 --delta 0.05 --gamma 0.9
