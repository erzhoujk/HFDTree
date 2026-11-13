#!/bin/bash
set -euo pipefail

GPU_ARGS=()
IFS=',' read -ra ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?gflow did not allocate GPUs}"
for gpu in "${ALLOCATED_GPUS[@]}"; do
  GPU_ARGS+=(--device "nvidia.com/gpu=${gpu}")
done

IMAGE=${HFDTREE_CUDA_CHECK_IMAGE:-docker.m.daocloud.io/nvidia/cuda:12.4.0-base-ubuntu22.04}

echo "allocated_physical_gpus=${CUDA_VISIBLE_DEVICES}"
docker info --format 'rootless={{json .SecurityOptions}} driver={{.Driver}}'
docker run --rm "${GPU_ARGS[@]}" "$IMAGE" nvidia-smi -L
