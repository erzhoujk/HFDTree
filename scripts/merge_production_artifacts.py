#!/usr/bin/env python3
"""Merge collection/judge/score shards and report eligibility and resource cost."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def _merge(pattern: str, directory: Path, output: Path, key_fields: tuple[str, ...]):
    rows = []
    seen = set()
    files = sorted(directory.glob(pattern))
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = tuple(row[field] for field in key_fields)
            if key in seen:
                raise ValueError(f"duplicate key {key} in {path}:{line_number}")
            seen.add(key)
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return rows, files


def _sum_metrics(directory: Path):
    result = Counter()
    for path in directory.glob("metrics_*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        result["wall_seconds_sum"] += float(value.get("wall_seconds", 0))
        policy = value.get("policy", value.get("judge", value.get("scorer", {})))
        for name in (
            "input_tokens",
            "output_tokens",
            "scored_prompt_tokens",
            "scored_action_tokens",
            "forward_calls",
        ):
            result[name] += int(policy.get(name, 0))
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=206)
    args = parser.parse_args()
    collections, collection_files = _merge(
        "rollouts_*.jsonl",
        args.collection_dir,
        args.output_dir / "rollouts.jsonl",
        ("task_id",),
    )
    judged, judge_files = _merge(
        "judged_*.jsonl", args.judge_dir, args.output_dir / "judged.jsonl", ("task_id",)
    )
    verified, _ = _merge(
        "verified_*.jsonl",
        args.judge_dir,
        args.output_dir / "verified_pairs.jsonl",
        ("task_id",),
    )
    nodes, score_files = _merge(
        "nodes_*.jsonl",
        args.score_dir,
        args.output_dir / "scored_nodes.jsonl",
        ("task_id", "node_id"),
    )
    if len(collections) != args.expected_tasks:
        raise ValueError(f"expected {args.expected_tasks} collection tasks, found {len(collections)}")
    eligible_roles = Counter()
    collection_by_task = {row["task_id"]: row for row in collections}
    for pair in verified:
        eligible_roles[collection_by_task[pair["task_id"]]["manifest_metadata"]["source_role"]] += 1
    report = {
        "collection_shards": len(collection_files),
        "judge_shards": len(judge_files),
        "score_shards": len(score_files),
        "collected_tasks": len(collections),
        "judged_tasks": len(judged),
        "verified_pairs": len(verified),
        "pair_eligibility_rate": len(verified) / len(collections) if collections else 0,
        "eligible_by_source_role": dict(eligible_roles),
        "scored_training_nodes": len(nodes),
        "collection_cost": _sum_metrics(args.collection_dir),
        "judge_cost": _sum_metrics(args.judge_dir),
        "scoring_cost": _sum_metrics(args.score_dir),
        "environment_executions": sum(
            len(rollout["runtime_trace"])
            for record in collections
            for rollout in record["rollouts"]
        ),
        "reference_environment_executions": sum(
            len(trajectory["runtime_trace"])
            for record in collections
            for trajectory in record["reference_candidates"].values()
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "production_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
