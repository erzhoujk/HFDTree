#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=/home/zhoujiankui/anaconda3/envs/opd-verl/bin/python
ARTIFACTS=artifacts/production/qwen3_4b_g8_v2

"$PYTHON" scripts/merge_production_artifacts.py \
  --collection-dir "$ARTIFACTS/train_collection" \
  --judge-dir "$ARTIFACTS/train_judged" \
  --score-dir "$ARTIFACTS/train_scored" \
  --output-dir "$ARTIFACTS/merged"

"$PYTHON" scripts/validate_trajectory_pairs.py \
  "$ARTIFACTS/merged/verified_pairs.jsonl" \
  > "$ARTIFACTS/merged/pair_validation.json"

"$PYTHON" scripts/build_bireference_control.py \
  --rollouts "$ARTIFACTS/merged/rollouts.jsonl" \
  --verified-pairs "$ARTIFACTS/merged/verified_pairs.jsonl" \
  --benchmark-root external/Agent-SafetyBench \
  --output "$ARTIFACTS/merged/direct_bireference_nodes.jsonl"

"$PYTHON" scripts/build_matched_bireference_control.py \
  --direct-bireference "$ARTIFACTS/merged/direct_bireference_nodes.jsonl" \
  --scored-nodes "$ARTIFACTS/merged/scored_nodes.jsonl" \
  --output "$ARTIFACTS/merged/direct_bireference_matched_fallback_nodes.jsonl" \
  --delta 0.05 --epsilon 0.05

cat > "$ARTIFACTS/merged/training_budget.json" <<'JSON'
{
  "model": "/home/zhoujiankui/LLM_MODEL/Qwen/Qwen3-4B",
  "optimizer": "AdamW",
  "learning_rate": 1e-5,
  "weight_decay": 0.01,
  "optimizer_steps": 1000,
  "batch_size": 2,
  "gradient_accumulation_steps": 8,
  "effective_batch_size": 16,
  "max_length": 4096,
  "lora_rank": 16,
  "lora_alpha": 32,
  "seed": 42
}
JSON
