#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SHARD_INDEX=${SHARD_INDEX:-${GFLOW_ARRAY_TASK_ID:-0}}
COLLECTION_ROOT=${COLLECTION_ROOT:-artifacts/production/qwen3_4b_g8_v2/train_collection}
JUDGE_ROOT=${JUDGE_ROOT:-artifacts/production/qwen3_4b_g8_v2/train_judged}
mkdir -p "$ROOT/$JUDGE_ROOT"
SHARD=$(printf '%03d' "$SHARD_INDEX")

bash "$ROOT/scripts/docker/run_gpu_command.sh" \
  /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
  scripts/judge_agent_safety_rollouts.py \
  --input "$COLLECTION_ROOT/rollouts_${SHARD}.jsonl" \
  --judge-model /home/zhoujiankui/LLM_MODEL/Qwen/Qwen2.5-7B-Instruct \
  --judged-output "$JUDGE_ROOT/judged_${SHARD}.jsonl" \
  --verified-output "$JUDGE_ROOT/verified_${SHARD}.jsonl" \
  --metrics-output "$JUDGE_ROOT/metrics_${SHARD}.json" \
  --minimum-confidence 0.7 --seed 7300
