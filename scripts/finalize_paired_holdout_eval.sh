#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
/home/zhoujiankui/anaconda3/bin/python scripts/aggregate_paired_holdout.py \
  --baseline artifacts/production/qwen3_4b_g8_v2/holdout_eval/base/judged \
  --adapter artifacts/production/qwen3_4b_g8_v2/holdout_eval/adapter/judged \
  --output artifacts/production/qwen3_4b_g8_v2/holdout_eval/paired_metrics.json
