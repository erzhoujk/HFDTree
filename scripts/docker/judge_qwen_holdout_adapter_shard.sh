#!/bin/bash
set -euo pipefail
EVAL_MODE=adapter bash "$(dirname "${BASH_SOURCE[0]}")/judge_qwen_holdout_shard.sh"
