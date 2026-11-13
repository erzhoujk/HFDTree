#!/usr/bin/env python3
"""Build rollout-collection seeds without inventing trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hfdtree.trajectory_data import build_execution_manifests, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", default="hfdtree-v1")
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    execution, evaluation = build_execution_manifests(
        artifact, train_fraction=args.train_fraction, seed=args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(execution, args.output_dir / "agent_safety_bench_execution_seeds.jsonl")
    write_jsonl(evaluation, args.output_dir / "agentharm_evaluation_only_index.jsonl")
    summary = {
        "AgentSafetyBench_execution_seeds": len(execution),
        "AgentSafetyBench_pair_groups": len({row["pair_group_id"] for row in execution}),
        "split_seed_counts": {
            split: sum(row["split"] == split for row in execution)
            for split in ("development_train", "development_holdout")
        },
        "split_pair_group_counts": {
            split: len(
                {
                    row["pair_group_id"]
                    for row in execution
                    if row["split"] == split
                }
            )
            for split in ("development_train", "development_holdout")
        },
        "source_record_overlap_between_splits": len(
            {
                row["source_record"]["id"]
                for row in execution
                if row["split"] == "development_train"
            }
            & {
                row["source_record"]["id"]
                for row in execution
                if row["split"] == "development_holdout"
            }
        ),
        "AgentHarm_evaluation_only_pairs": len(evaluation),
        "verified_trajectory_pairs_created": 0,
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
