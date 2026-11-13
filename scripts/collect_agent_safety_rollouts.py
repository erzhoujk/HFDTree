#!/usr/bin/env python3
"""Collect checkpoint-branched Agent-SafetyBench traces with a local model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from hfdtree.agent_safety_runtime import AgentSafetyTask
from hfdtree.production_rollout import ExplicitBranchCollector, TransformersPolicy


def _load_existing(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(int(json.loads(line)["source_record_id"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--split", default="development_train")
    parser.add_argument("--source-role", choices=("positive_sample", "negative_sample", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--rollout-width", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()

    selected = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != args.split:
                continue
            if args.source_role != "all" and row["source_role_in_similarity_pair"] != args.source_role:
                continue
            selected.append(row)
    selected.sort(key=lambda row: int(row["source_record"]["id"]))
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    selected = [
        row for index, row in enumerate(selected) if index % args.num_shards == args.shard_index
    ]
    if args.limit > 0:
        selected = selected[: args.limit]
    completed = _load_existing(args.output)
    selected = [row for row in selected if int(row["source_record"]["id"]) not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = TransformersPolicy(args.model, adapter_path=args.adapter)
    collector = ExplicitBranchCollector(
        policy,
        max_turns=args.max_turns,
        branch_factor=args.branch_factor,
        rollout_width=args.rollout_width,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    start = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as output:
        for index, row in enumerate(selected):
            task = AgentSafetyTask.from_record(row["source_record"])
            record = collector.collect_task(
                task,
                benchmark_root=str(args.benchmark_root),
                seed=args.seed + index * 1_000_003,
                include_references=not args.evaluation_only,
            )
            record["manifest_metadata"] = {
                "pair_group_id": row["pair_group_id"],
                "split": row["split"],
                "source_role": row["source_role_in_similarity_pair"],
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            print(f"collected {index + 1}/{len(selected)} task_id={task.id}", flush=True)
    metrics = {
        "wall_seconds": time.perf_counter() - start,
        "tasks_requested_this_run": len(selected),
        "tasks_previously_completed": len(completed),
        "policy": policy.metrics(),
        "configuration": vars(args) | {"manifest": str(args.manifest), "benchmark_root": str(args.benchmark_root), "output": str(args.output), "metrics_output": str(args.metrics_output)},
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
