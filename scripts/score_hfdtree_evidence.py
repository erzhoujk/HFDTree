#!/usr/bin/env python3
"""Score exact action tokens and emit all controlled credit variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from hfdtree.agent_safety_runtime import AgentSafetyRuntime, AgentSafetyTask
from hfdtree.evidence_pipeline import FixedActionScorer, score_task_rollouts


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--verified-pairs", type=Path, required=True)
    parser.add_argument("--judged-rollouts", type=Path)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()

    collections = {record["task_id"]: record for record in _jsonl(args.rollouts)}
    pairs = {record["task_id"]: record for record in _jsonl(args.verified_pairs)}
    judged = (
        {record["task_id"]: record for record in _jsonl(args.judged_rollouts)}
        if args.judged_rollouts
        else {}
    )
    common = sorted(collections.keys() & pairs.keys())
    if not common:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps({"wall_seconds": 0.0, "tasks": 0, "training_nodes": 0, "scorer": {}})
            + "\n",
            encoding="utf-8",
        )
        print("no verified pairs in this shard", flush=True)
        return
    scorer = FixedActionScorer(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    examples = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, task_id in enumerate(common):
            collection = collections[task_id]
            task = AgentSafetyTask.from_record(collection["source_record"])
            runtime = AgentSafetyRuntime(task, benchmark_root=args.benchmark_root)
            outcome_by_id = None
            if task_id in judged:
                outcome_by_id = {
                    str(item["trajectory"]["trajectory_id"]): item["judgment"]
                    for item in judged[task_id].get("judged_trajectories", [])
                    if item["trajectory"].get("candidate_role") == "unprivileged_rollout"
                }
            rows = score_task_rollouts(
                scorer,
                collection,
                pairs[task_id],
                runtime.available_tools(),
                outcome_by_trajectory_id=outcome_by_id,
                temperature=args.temperature,
                delta=args.delta,
                gamma=args.gamma,
            )
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            examples += len(rows)
            print(f"scored {index + 1}/{len(common)} task={task_id} nodes={len(rows)}", flush=True)
    metrics = {
        "wall_seconds": time.perf_counter() - start,
        "tasks": len(common),
        "training_nodes": examples,
        "configuration": {
            "temperature": args.temperature,
            "delta": args.delta,
            "gamma": args.gamma,
        },
        "scorer": scorer.metrics(),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
