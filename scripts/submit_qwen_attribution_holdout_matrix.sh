#!/bin/bash
# Submit every method required by the matched-fallback attribution table.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PHASE=${1:-}
SEED=${2:-42}
[[ "$PHASE" == collect || "$PHASE" == judge ]] || {
  echo "usage: $0 {collect|judge} [SEED]" >&2
  exit 2
}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer" >&2; exit 2; }

VARIANTS=(
  full
  no_fallback
  propagation_without_centering_matched_fallback
  two_sided_local_matched_fallback
  one_sided_tree_matched_fallback
  outcome_tree_matched_fallback
  direct_bireference_matched_fallback
)

for variant in "${VARIANTS[@]}"; do
  bash scripts/submit_qwen_controlled_holdout.sh "$PHASE" "$variant" "$SEED"
done
