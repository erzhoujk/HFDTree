#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

bash "$ROOT/scripts/docker/run_gpu_command.sh" \
  /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
  scripts/train_hfdtree_lora.py \
  --data artifacts/smoke/qwen3_4b_scored_nodes_v3.jsonl \
  --model /home/zhoujiankui/LLM_MODEL/Qwen/Qwen3-4B \
  --output-dir artifacts/smoke/qwen3_4b_full_lora_v3 \
  --variant full --seed 42 --batch-size 1 --gradient-accumulation-steps 1 \
  --max-steps 2 --max-length 2048 --lora-rank 8 --lora-alpha 16
