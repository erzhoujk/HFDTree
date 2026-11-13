#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
/home/zhoujiankui/anaconda3/bin/python scripts/summarize_qwen_production_training.py \
  artifacts/production/qwen3_4b_g8_v2/training \
  --output artifacts/production/qwen3_4b_g8_v2/training/variant_summary.json
