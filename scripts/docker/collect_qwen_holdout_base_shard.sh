#!/bin/bash
set -euo pipefail
EVAL_MODE=base bash "$(dirname "${BASH_SOURCE[0]}")/collect_qwen_holdout_shard.sh"
