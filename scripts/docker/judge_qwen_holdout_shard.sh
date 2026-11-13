#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MODE=${EVAL_MODE:?set EVAL_MODE to base or adapter}
EVAL_LABEL=${HFDTREE_EVAL_LABEL:-$MODE}
SHARD_INDEX=${GFLOW_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}
BASE=artifacts/production/qwen3_4b_g8_v2/holdout_eval
if [[ ! "$EVAL_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "HFDTREE_EVAL_LABEL may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
SHARD=$(printf '%03d' "$SHARD_INDEX")
OUT="$BASE/$EVAL_LABEL/judged"
mkdir -p "$OUT"

bash scripts/docker/run_gpu_command.sh \
  /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
  scripts/judge_agent_safety_rollouts.py \
  --input "$BASE/$EVAL_LABEL/rollouts/rollouts_${SHARD}.jsonl" \
  --judge-model /home/zhoujiankui/LLM_MODEL/Qwen/Qwen2.5-7B-Instruct \
  --judged-output "$OUT/judged_${SHARD}.jsonl" \
  --verified-output "$OUT/verified_${SHARD}.jsonl" \
  --metrics-output "$OUT/metrics_${SHARD}.json" \
  --minimum-confidence 0.7 --seed $((99100 + SHARD_INDEX * 1000003))
