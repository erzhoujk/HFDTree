#!/bin/bash
# Submit one frozen adapter's 52-shard holdout evaluation through gflow.
# Usage:
#   bash scripts/submit_qwen_controlled_holdout.sh collect full
#   bash scripts/submit_qwen_controlled_holdout.sh judge full
#   bash scripts/submit_qwen_controlled_holdout.sh aggregate full 42
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PHASE=${1:-}
VARIANT=${2:-}
SEED=${3:-42}
if [[ -z "$PHASE" || -z "$VARIANT" ]]; then
  echo "usage: $0 {collect|judge|aggregate} VARIANT [SEED]" >&2
  exit 2
fi
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer" >&2; exit 2; }
case "$VARIANT" in
  full|no_fallback|propagation_without_centering|two_sided_local_only|two_sided_local_with_fallback|one_sided_tree|outcome_tree_equal_information|direct_bireference_equal_information|propagation_without_centering_matched_fallback|two_sided_local_matched_fallback|one_sided_tree_matched_fallback|outcome_tree_matched_fallback|direct_bireference_matched_fallback) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

ADAPTER="artifacts/production/qwen3_4b_g8_v2/training/${VARIANT}_seed${SEED}"
[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { echo "missing adapter: $ADAPTER" >&2; exit 2; }
LABEL="${VARIANT}_seed${SEED}"
BASE="artifacts/production/qwen3_4b_g8_v2/holdout_eval"

case "$PHASE" in
  collect)
    gbatch -g 1 --gpu-memory 36G --array 0-51 --max-concurrent 3 -n "hfdtree-${LABEL}-collect" \
      env EVAL_MODE=adapter HFDTREE_EVAL_LABEL="$LABEL" HFDTREE_ADAPTER="$ADAPTER" \
      bash scripts/docker/collect_qwen_holdout_shard.sh
    ;;
  judge)
    gbatch -g 1 --gpu-memory 36G --array 0-51 --max-concurrent 3 -n "hfdtree-${LABEL}-judge" \
      env EVAL_MODE=adapter HFDTREE_EVAL_LABEL="$LABEL" HFDTREE_ADAPTER="$ADAPTER" \
      bash scripts/docker/judge_qwen_holdout_shard.sh
    ;;
  aggregate)
    /home/zhoujiankui/anaconda3/bin/python scripts/aggregate_paired_holdout.py \
      --baseline "$BASE/base/judged" \
      --adapter "$BASE/$LABEL/judged" \
      --output "$BASE/${LABEL}_paired_metrics.json"
    ;;
  *) echo "phase must be collect, judge, or aggregate" >&2; exit 2 ;;
esac
