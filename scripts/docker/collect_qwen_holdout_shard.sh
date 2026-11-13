#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MODE=${EVAL_MODE:?set EVAL_MODE to base or adapter}
EVAL_LABEL=${HFDTREE_EVAL_LABEL:-$MODE}
SHARD_INDEX=${GFLOW_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}
NUM_SHARDS=52
BASE=artifacts/production/qwen3_4b_g8_v2/holdout_eval
MODEL=/home/zhoujiankui/LLM_MODEL/Qwen/Qwen3-4B
ADAPTER_ARGS=()
if [[ "$MODE" == adapter ]]; then
  ADAPTER=${HFDTREE_ADAPTER:-artifacts/production/qwen3_4b_g8_v2/training/full_seed42}
  ADAPTER_ARGS=(--adapter "$ADAPTER")
elif [[ "$MODE" != base ]]; then
  echo "EVAL_MODE must be base or adapter" >&2
  exit 2
fi
if [[ ! "$EVAL_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "HFDTREE_EVAL_LABEL may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
SHARD=$(printf '%03d' "$SHARD_INDEX")
TASK_SEED=$((99042 + SHARD_INDEX * 1000003))
OUT="$BASE/$EVAL_LABEL/rollouts"
mkdir -p "$OUT"

bash scripts/docker/run_gpu_command.sh \
  /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
  scripts/collect_agent_safety_rollouts.py \
  --manifest data/manifests/agent_safety_bench_execution_seeds.jsonl \
  --benchmark-root external/Agent-SafetyBench \
  --model "$MODEL" "${ADAPTER_ARGS[@]}" \
  --output "$OUT/rollouts_${SHARD}.jsonl" \
  --metrics-output "$OUT/metrics_${SHARD}.json" \
  --split development_holdout --source-role all \
  --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" \
  --seed "$TASK_SEED" --max-turns 6 --branch-factor 2 --rollout-width 8 \
  --temperature 0.8 --top-p 0.95 --max-new-tokens 256 --evaluation-only
