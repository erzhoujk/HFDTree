#!/bin/bash
# Train the controls needed to hold fallback information fixed.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
SEED=${1:-42}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer" >&2; exit 2; }

DATA=artifacts/production/qwen3_4b_g8_v2/merged/direct_bireference_matched_fallback_nodes.jsonl
[[ -f "$DATA" ]] || {
  echo "missing $DATA; run scripts/prepare_qwen_production_training.sh first" >&2
  exit 2
}

if [[ "$SEED" != 42 ]]; then
  gbatch -g 1 --gpu-memory 36G --array 0-1 --max-concurrent 2 \
    -t 02:00:00 -n "hfdtree-core-seed${SEED}-train" \
    env HFDTREE_SEED="$SEED" bash scripts/docker/train_qwen_production_variant.sh
fi

gbatch -g 1 --gpu-memory 36G --array 8-12 --max-concurrent 3 \
  -t 02:00:00 -n "hfdtree-matched-fallback-seed${SEED}-train" \
  env HFDTREE_SEED="$SEED" bash scripts/docker/train_qwen_production_variant.sh
