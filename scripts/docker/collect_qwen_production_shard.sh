#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SHARD_INDEX=${SHARD_INDEX:-${GFLOW_ARRAY_TASK_ID:-0}}
NUM_SHARDS=${NUM_SHARDS:-42}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/production/qwen3_4b_g8_v2/train_collection}
mkdir -p "$ROOT/$OUTPUT_ROOT"
SHARD=$(printf '%03d' "$SHARD_INDEX")

bash "$ROOT/scripts/docker/run_gpu_command.sh" \
  /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
  scripts/collect_agent_safety_rollouts.py \
  --manifest data/manifests/agent_safety_bench_execution_seeds.jsonl \
  --benchmark-root external/Agent-SafetyBench \
  --model /home/zhoujiankui/LLM_MODEL/Qwen/Qwen3-4B \
  --output "$OUTPUT_ROOT/rollouts_${SHARD}.jsonl" \
  --metrics-output "$OUTPUT_ROOT/metrics_${SHARD}.json" \
  --split development_train --source-role all \
  --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" \
  --seed 4200 --max-turns 6 --branch-factor 2 --rollout-width 8 \
  --temperature 0.8 --top-p 0.95 --max-new-tokens 256
