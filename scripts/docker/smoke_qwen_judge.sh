#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$ROOT/artifacts/smoke"

bash "$ROOT/scripts/docker/run_gpu_command.sh" \
  /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
  scripts/judge_agent_safety_rollouts.py \
  --input artifacts/smoke/qwen3_4b_rollouts_v2.jsonl \
  --judge-model /home/zhoujiankui/LLM_MODEL/Qwen/Qwen2.5-7B-Instruct \
  --judged-output artifacts/smoke/qwen3_4b_judged_v3.jsonl \
  --verified-output artifacts/smoke/qwen3_4b_verified_pairs_v3.jsonl \
  --metrics-output artifacts/smoke/qwen3_4b_judge_metrics_v3.json \
  --minimum-confidence 0.7 \
  --seed 73
