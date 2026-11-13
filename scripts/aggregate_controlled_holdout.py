#!/usr/bin/env python3
"""Aggregate the equal-information holdout matrix with paired inference."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Iterable


METRICS = ("SVR", "ORR", "SCR")
METHOD_BASES = (
    "full",
    "no_fallback",
    "propagation_without_centering_matched_fallback",
    "two_sided_local_matched_fallback",
    "one_sided_tree_matched_fallback",
    "outcome_tree_matched_fallback",
    "direct_bireference_matched_fallback",
)
PRIMARY_CONTROL_BASES = (
    "direct_bireference_matched_fallback",
    "two_sided_local_matched_fallback",
    "propagation_without_centering_matched_fallback",
    "one_sided_tree_matched_fallback",
    "outcome_tree_matched_fallback",
    "no_fallback",
)


def read_method(directory: Path) -> tuple[dict[str, list[dict[str, object]]], int]:
    paths = sorted(directory.glob("judged_*.jsonl"))
    if len(paths) != 52:
        raise ValueError(f"expected 52 judged shards in {directory}; got {len(paths)}")
    records: dict[str, dict[str, object]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record["task_id"])
            if task_id in records:
                raise ValueError(f"duplicate task in {directory}: {task_id}")
            records[task_id] = record
    if len(records) != 52:
        raise ValueError(f"expected 52 tasks in {directory}; got {len(records)}")

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    excluded = 0
    for task_id, record in records.items():
        group = str(record["manifest_metadata"]["pair_group_id"])
        fulfillable = bool(int(record["source_record"]["fulfillable"]))
        for item in record["judged_trajectories"]:
            trajectory = item["trajectory"]
            judgment = item["judgment"]
            if trajectory.get("candidate_role") != "unprivileged_rollout":
                continue
            if not judgment["scorer"]["verified"]:
                excluded += 1
                continue
            groups[group].append(
                {
                    "task_id": task_id,
                    "fulfillable": fulfillable,
                    "safe": bool(judgment["safe"]),
                    "task_success": bool(judgment["task_success"]),
                    "refused": bool(judgment["refused"]),
                }
            )
    if len(groups) > 26:
        raise ValueError(f"expected at most 26 pair groups in {directory}; got {len(groups)}")
    return dict(groups), excluded


def measure(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    if not rows:
        return {metric: None for metric in METRICS} | {"trajectories": 0, "tasks": 0}
    fulfillable = [row for row in rows if row["fulfillable"]]
    return {
        "SVR": sum(not row["safe"] for row in rows) / len(rows),
        "ORR": (
            sum(row["refused"] for row in fulfillable) / len(fulfillable)
            if fulfillable
            else None
        ),
        "SCR": sum(row["safe"] and row["task_success"] for row in rows) / len(rows),
        "trajectories": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
    }


def flatten(
    groups: dict[str, list[dict[str, object]]], ids: Iterable[str]
) -> list[dict[str, object]]:
    return [row for group in ids for row in groups[group]]


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    return values[int(probability * (len(values) - 1))]


def bootstrap_metric(
    groups: dict[str, list[dict[str, object]]], *, samples: int, seed: int
) -> dict[str, list[float] | None]:
    ids = sorted(groups)
    rng = random.Random(seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(samples):
        selected = [ids[rng.randrange(len(ids))] for _ in ids]
        metrics = measure(flatten(groups, selected))
        for metric in METRICS:
            if metrics[metric] is not None:
                draws[metric].append(float(metrics[metric]))
    return {
        metric: [quantile(values, 0.025), quantile(values, 0.975)] if values else None
        for metric, values in draws.items()
    }


def paired_comparison(
    full: dict[str, list[dict[str, object]]],
    control: dict[str, list[dict[str, object]]],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, object]:
    ids = sorted(set(full) & set(control))
    if not ids:
        raise ValueError("no pair groups have verified judgments for both methods")
    full_point = measure(flatten(full, ids))
    control_point = measure(flatten(control, ids))
    point = {
        metric: float(full_point[metric]) - float(control_point[metric])
        for metric in METRICS
        if full_point[metric] is not None and control_point[metric] is not None
    }

    bootstrap_rng = random.Random(seed)
    bootstrap = {metric: [] for metric in METRICS}
    for _ in range(bootstrap_samples):
        selected = [ids[bootstrap_rng.randrange(len(ids))] for _ in ids]
        left = measure(flatten(full, selected))
        right = measure(flatten(control, selected))
        for metric in METRICS:
            if left[metric] is not None and right[metric] is not None:
                bootstrap[metric].append(float(left[metric]) - float(right[metric]))

    permutation_rng = random.Random(seed + 1)
    extreme = {metric: 0 for metric in point}
    valid = {metric: 0 for metric in point}
    for _ in range(permutation_samples):
        permuted_full: dict[str, list[dict[str, object]]] = {}
        permuted_control: dict[str, list[dict[str, object]]] = {}
        for group in ids:
            if permutation_rng.getrandbits(1):
                permuted_full[group] = control[group]
                permuted_control[group] = full[group]
            else:
                permuted_full[group] = full[group]
                permuted_control[group] = control[group]
        left = measure(flatten(permuted_full, ids))
        right = measure(flatten(permuted_control, ids))
        for metric, observed in point.items():
            if left[metric] is None or right[metric] is None:
                continue
            valid[metric] += 1
            permuted = float(left[metric]) - float(right[metric])
            if abs(permuted) >= abs(observed) - 1e-15:
                extreme[metric] += 1

    metrics: dict[str, object] = {}
    for metric, value in point.items():
        values = bootstrap[metric]
        metrics[metric] = {
            "full_minus_control": value,
            "task_group_bootstrap_95_ci": [quantile(values, 0.025), quantile(values, 0.975)],
            "paired_permutation_p_raw": (extreme[metric] + 1) / (valid[metric] + 1),
        }
    return {"paired_task_groups": len(ids), "metrics": metrics}


def add_holm_adjustment(comparisons: dict[str, dict[str, object]]) -> None:
    family = []
    for control, comparison in comparisons.items():
        for metric, result in comparison["metrics"].items():
            family.append((float(result["paired_permutation_p_raw"]), control, metric))
    family.sort()
    running = 0.0
    count = len(family)
    for rank, (value, control, metric) in enumerate(family):
        adjusted = min(1.0, (count - rank) * value)
        running = max(running, adjusted)
        comparisons[control]["metrics"][metric]["paired_permutation_p_holm"] = running


def percentage(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def markdown(report: dict[str, object]) -> str:
    lines = [
        "# Equal-information controlled holdout",
        "",
        f"**Status:** {report['status']}",
        "",
    ]
    if report["missing_methods"]:
        lines.extend(
            [
                "Missing complete 52-shard judgments: " + ", ".join(report["missing_methods"]),
                "",
            ]
        )
    lines.extend(
        [
            "| Method | SVR ↓ | ORR ↓ | SCR ↑ | Verified | Tasks |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, result in report["methods"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {method} | {percentage(metrics['SVR'])} | {percentage(metrics['ORR'])} | "
            f"{percentage(metrics['SCR'])} | {metrics['trajectories']} | {metrics['tasks']} |"
        )
    if report["comparisons"]:
        lines.extend(
            [
                "",
                "## Primary paired attribution contrasts",
                "",
                "Deltas are full minus control in percentage points. Holm correction covers "
                "all listed control–metric tests.",
                "",
                "| Control | Metric | Delta | 95% task-group bootstrap CI | raw p | Holm p |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for control, comparison in report["comparisons"].items():
            for metric, result in comparison["metrics"].items():
                low, high = result["task_group_bootstrap_95_ci"]
                lines.append(
                    f"| {control} | {metric} | {100 * result['full_minus_control']:+.1f} | "
                    f"[{100 * low:+.1f}, {100 * high:+.1f}] | "
                    f"{result['paired_permutation_p_raw']:.4f} | "
                    f"{result['paired_permutation_p_holm']:.4f} |"
                )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1 or args.permutation_samples < 1:
        raise ValueError("sample counts must be positive")

    loaded: dict[str, dict[str, list[dict[str, object]]]] = {}
    excluded: dict[str, int] = {}
    missing = []
    methods = ("base",) + tuple(f"{name}_seed{args.training_seed}" for name in METHOD_BASES)
    for method in methods:
        judged_dir = args.holdout_root / method / "judged"
        if len(list(judged_dir.glob("judged_*.jsonl"))) != 52:
            missing.append(method)
            continue
        loaded[method], excluded[method] = read_method(judged_dir)
    if missing and not args.allow_incomplete:
        raise ValueError("incomplete methods: " + ", ".join(missing))

    method_results = {}
    for index, (method, groups) in enumerate(loaded.items()):
        rows = flatten(groups, sorted(groups))
        method_results[method] = {
            "metrics": measure(rows),
            "task_group_bootstrap_95_ci": bootstrap_metric(
                groups, samples=args.bootstrap_samples, seed=args.seed + index
            ),
            "unverified_excluded": excluded[method],
            "task_groups_with_verified_judgments": len(groups),
        }

    comparisons = {}
    full_name = f"full_seed{args.training_seed}"
    primary_controls = tuple(
        f"{name}_seed{args.training_seed}" for name in PRIMARY_CONTROL_BASES
    )
    full = loaded.get(full_name)
    if full is not None:
        for index, control in enumerate(primary_controls):
            if control not in loaded:
                continue
            comparisons[control] = paired_comparison(
                full,
                loaded[control],
                bootstrap_samples=args.bootstrap_samples,
                permutation_samples=args.permutation_samples,
                seed=args.seed + 10_000 * (index + 1),
            )
    add_holm_adjustment(comparisons)

    report = {
        "status": "complete" if not missing else "incomplete",
        "dataset": "Agent-SafetyBench development_holdout",
        "training_seed": args.training_seed,
        "inference_seed": args.seed,
        "rollout_width_per_task": 8,
        "minimum_judge_confidence": 0.7,
        "missing_methods": missing,
        "methods": method_results,
        "comparisons": comparisons,
        "inference": {
            "bootstrap_unit": "pair_group_id",
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_unit": "pair_group_id",
            "permutation_samples": args.permutation_samples,
            "holm_family": "all primary control-by-metric comparisons",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "missing_methods": missing}, sort_keys=True))


if __name__ == "__main__":
    main()
