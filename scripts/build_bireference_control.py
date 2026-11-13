#!/usr/bin/env python3
"""Build a direct bi-reference control from the same verified trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hfdtree.agent_safety_runtime import AgentSafetyRuntime, AgentSafetyTask


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--verified-pairs", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collections = {row["task_id"]: row for row in _jsonl(args.rollouts)}
    pairs = {row["task_id"]: row for row in _jsonl(args.verified_pairs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for task_id in sorted(collections.keys() & pairs.keys()):
            task = AgentSafetyTask.from_record(collections[task_id]["source_record"])
            runtime = AgentSafetyRuntime(task, benchmark_root=args.benchmark_root)
            tools = runtime.available_tools()
            for role, weight in (("positive_trajectory", 1.0), ("negative_trajectory", -1.0)):
                for turn in pairs[task_id][role]["turns"]:
                    row = {
                        "task_id": task_id,
                        "source_record_id": task.id,
                        "control": "direct_bireference_equal_information",
                        "reference_role": role,
                        "prefix_messages": turn["prefix_messages"],
                        "tools": tools,
                        "action": turn["action"],
                        "observation": turn["observation"],
                        "action_token_ids": turn["action_token_ids"],
                        "normalized_evidence": 1.0,
                        "reference_valid": False,
                        "reference_action_token_ids": [],
                        "credit_full": weight,
                        "credit_propagation_without_centering": weight,
                        "credit_two_sided_local_only": weight,
                        "credit_one_sided_tree": weight,
                        "credit_outcome_tree": weight,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
    print(json.dumps({"tasks": len(collections.keys() & pairs.keys()), "turns": count}))


if __name__ == "__main__":
    main()
