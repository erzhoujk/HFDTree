#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: run_gpu_command.sh COMMAND [ARG ...]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=${HFDTREE_IMAGE:-wan21-t2vsafetybench:1.1}
CONDA_ENV=${HFDTREE_CONDA_ENV:-/home/zhoujiankui/anaconda3/envs/opd-verl}
MODEL_ROOT=${HFDTREE_MODEL_ROOT:-/home/zhoujiankui/LLM_MODEL}

GPU_ARGS=()
IFS=',' read -ra ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?gflow did not allocate GPUs}"
for gpu in "${ALLOCATED_GPUS[@]}"; do
  GPU_ARGS+=(--device "nvidia.com/gpu=${gpu}")
done

COMMAND_JSON=$(
  /home/zhoujiankui/anaconda3/bin/python -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
)

docker run --rm \
  "${GPU_ARGS[@]}" \
  --ipc=host \
  --ulimit stack=67108864 \
  -v "$ROOT:/workspace/HFDTree" \
  -v "$CONDA_ENV:$CONDA_ENV:ro" \
  -v "$MODEL_ROOT:$MODEL_ROOT:ro" \
  -w /workspace/HFDTree \
  -e PYTHONPATH=/workspace/HFDTree/src \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e HFDTREE_COMMAND_JSON="$COMMAND_JSON" \
  "$IMAGE" \
  "$CONDA_ENV/bin/python" -c \
  'import json,os,subprocess,sys; raise SystemExit(subprocess.call(json.loads(os.environ["HFDTREE_COMMAND_JSON"])))'
