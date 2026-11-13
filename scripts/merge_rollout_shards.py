#!/usr/bin/env python3
"""Merge and validate unique task records from production rollout shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int)
    args = parser.parse_args()
    records = {}
    files = sorted(args.input_dir.glob("rollouts_*.jsonl"))
    if not files:
        raise ValueError("no rollout shard files found")
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record["task_id"])
            if task_id in records:
                raise ValueError(f"duplicate task {task_id} in {path}:{line_number}")
            records[task_id] = record
    if args.expected_tasks is not None and len(records) != args.expected_tasks:
        raise ValueError(f"expected {args.expected_tasks} tasks, found {len(records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task_id in sorted(records, key=lambda value: int(value.rsplit("_", 1)[-1])):
            handle.write(json.dumps(records[task_id], ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"shards": len(files), "tasks": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
