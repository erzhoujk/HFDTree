#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$ROOT/artifacts/smoke"

bash "$ROOT/scripts/docker/run_gpu_command.sh" \
  /home/zhoujiankui/anaconda3/envs/opd-verl/bin/python \
  scripts/collect_agent_safety_rollouts.py \
  --manifest data/manifests/agent_safety_bench_execution_seeds.jsonl \
  --benchmark-root external/Agent-SafetyBench \
  --model /home/zhoujiankui/LLM_MODEL/Qwen/Qwen3-4B \
  --output artifacts/smoke/qwen3_4b_rollouts_v2.jsonl \
  --metrics-output artifacts/smoke/qwen3_4b_rollout_metrics_v2.json \
  --split development_train \
  --source-role negative_sample \
  --limit 1 \
  --seed 42 \
  --max-turns 3 \
  --branch-factor 2 \
  --rollout-width 4 \
  --max-new-tokens 128
