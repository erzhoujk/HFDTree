#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SHARD_INDEX=${SHARD_INDEX:-${GFLOW_ARRAY_TASK_ID:-0}}
COLLECTION_ROOT=${COLLECTION_ROOT:-artifacts/production/qwen3_4b_g8_v2/train_collection}
JUDGE_ROOT=${JUDGE_ROOT:-artifacts/production/qwen3_4b_g8_v2/train_judged}
SCORE_ROOT=${SCORE_ROOT:-artifacts/production/qwen3_4b_g8_v2/train_scored}
mkdir -p "$ROOT/$SCORE_ROOT"
SHARD=$(printf '%03d' "$SHARD_INDEX")

bash "$ROOT/scripts/docker/run_gpu_command.sh" \
  /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
  scripts/score_hfdtree_evidence.py \
  --rollouts "$COLLECTION_ROOT/rollouts_${SHARD}.jsonl" \
  --verified-pairs "$JUDGE_ROOT/verified_${SHARD}.jsonl" \
  --judged-rollouts "$JUDGE_ROOT/judged_${SHARD}.jsonl" \
  --benchmark-root external/Agent-SafetyBench \
  --model /home/zhoujiankui/LLM_MODEL/Qwen/Qwen3-4B \
  --output "$SCORE_ROOT/nodes_${SHARD}.jsonl" \
  --metrics-output "$SCORE_ROOT/metrics_${SHARD}.json" \
  --temperature 0.5 --delta 0.05 --gamma 0.9
