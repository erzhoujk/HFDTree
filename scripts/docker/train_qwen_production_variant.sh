#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
TASK_INDEX=${GFLOW_ARRAY_TASK_ID:-${TASK_INDEX:-0}}
SEED=${HFDTREE_SEED:-42}
ARTIFACTS=artifacts/production/qwen3_4b_g8_v2
MODEL=/home/zhoujiankui/LLM_MODEL/Qwen/Qwen3-4B
VARIANTS=(full no_fallback propagation_without_centering two_sided_local_only two_sided_local_with_fallback one_sided_tree outcome_tree_equal_information direct_bireference_equal_information propagation_without_centering_matched_fallback two_sided_local_matched_fallback one_sided_tree_matched_fallback outcome_tree_matched_fallback direct_bireference_matched_fallback)
VARIANT=${VARIANTS[$TASK_INDEX]:?unknown training array index: $TASK_INDEX}
DATA="$ARTIFACTS/merged/scored_nodes.jsonl"
if [[ "$VARIANT" == direct_bireference_equal_information ]]; then
  VARIANT=full
  DATA="$ARTIFACTS/merged/direct_bireference_nodes.jsonl"
elif [[ "$VARIANT" == direct_bireference_matched_fallback ]]; then
  DATA="$ARTIFACTS/merged/direct_bireference_matched_fallback_nodes.jsonl"
fi
OUTPUT="$ARTIFACTS/training/${VARIANTS[$TASK_INDEX]}_seed${SEED}"

TOTAL_STEPS=1000
CHUNK_STEPS=${HFDTREE_TRAINING_CHUNK_STEPS:-30}
TARGET_STEPS=$CHUNK_STEPS
while (( TARGET_STEPS < TOTAL_STEPS )); do
  ARGS=(
    scripts/train_hfdtree_lora.py
    --data "$DATA"
    --model "$MODEL"
    --output-dir "$OUTPUT"
    --variant "$VARIANT"
    --seed "$SEED"
    --learning-rate 1e-5
    --weight-decay 0.01
    --batch-size 2
    --gradient-accumulation-steps 8
    --max-steps "$TARGET_STEPS"
    --max-length 4096
    --lora-rank 16
    --lora-alpha 32
    --delta 0.05
    --epsilon 0.05
  )
  if [[ -f "$OUTPUT/training_state.pt" ]]; then
    ARGS+=(--resume)
  fi
  bash scripts/docker/run_gpu_command.sh \
    /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
    "${ARGS[@]}"
  TARGET_STEPS=$((TARGET_STEPS + CHUNK_STEPS))
done

if (( TARGET_STEPS - CHUNK_STEPS != TOTAL_STEPS )); then
  ARGS=(
    scripts/train_hfdtree_lora.py
    --data "$DATA"
    --model "$MODEL"
    --output-dir "$OUTPUT"
    --variant "$VARIANT"
    --seed "$SEED"
    --learning-rate 1e-5
    --weight-decay 0.01
    --batch-size 2
    --gradient-accumulation-steps 8
    --max-steps "$TOTAL_STEPS"
    --max-length 4096
    --lora-rank 16
    --lora-alpha 32
    --delta 0.05
    --epsilon 0.05
    --resume
  )
  bash scripts/docker/run_gpu_command.sh \
    /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
    "${ARGS[@]}"
fi
