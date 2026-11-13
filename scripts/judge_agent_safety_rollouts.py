#!/usr/bin/env python3
"""Judge collected traces and emit only same-state verified trajectory pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from hfdtree.production_rollout import TransformersPolicy
from hfdtree.trajectory_data import validate_verified_trajectory_pair
from hfdtree.trajectory_judge import TrajectoryJudge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judged-output", type=Path, required=True)
    parser.add_argument("--verified-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    policy = TransformersPolicy(args.judge_model)
    judge = TrajectoryJudge(policy, min_confidence=args.minimum_confidence)
    args.judged_output.parent.mkdir(parents=True, exist_ok=True)
    args.verified_output.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    start = time.perf_counter()
    with args.judged_output.open("w", encoding="utf-8") as judged_handle, args.verified_output.open("w", encoding="utf-8") as verified_handle:
        for index, record in enumerate(records):
            judged = judge.judge_collection(record, seed=args.seed + index * 100_003)
            judged_handle.write(json.dumps(judged, ensure_ascii=False, sort_keys=True) + "\n")
            pair = judge.select_verified_pair(judged)
            if pair is not None:
                validate_verified_trajectory_pair(pair)
                verified_handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")
                accepted += 1
            judged_handle.flush()
            verified_handle.flush()
            print(f"judged {index + 1}/{len(records)} accepted={accepted}", flush=True)
    metrics = {
        "wall_seconds": time.perf_counter() - start,
        "tasks": len(records),
        "verified_pairs": accepted,
        "eligibility_rate": accepted / len(records) if records else 0.0,
        "minimum_confidence": args.minimum_confidence,
        "judge": policy.metrics(),
    }
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
