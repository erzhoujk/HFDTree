#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/external"

if [ ! -d "$ROOT/external/Agent-SafetyBench/.git" ]; then
  git clone https://github.com/thu-coai/Agent-SafetyBench.git "$ROOT/external/Agent-SafetyBench"
fi
git -C "$ROOT/external/Agent-SafetyBench" checkout 74feea8de601b3a1449a93fcf70017fe61556f73

if [ ! -d "$ROOT/external/AgentHarm/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/ai-safety-institute/AgentHarm "$ROOT/external/AgentHarm"
fi
git -C "$ROOT/external/AgentHarm" checkout e23b3fe60a0da9037314b88e5ee3a0c054970dad

printf 'Agent-SafetyBench %s\n' "$(git -C "$ROOT/external/Agent-SafetyBench" rev-parse HEAD)"
printf 'AgentHarm %s\n' "$(git -C "$ROOT/external/AgentHarm" rev-parse HEAD)"
