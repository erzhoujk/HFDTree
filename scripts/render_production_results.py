#!/usr/bin/env python3
"""Render a reviewer-facing production experiment report from raw artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any


METRICS = ("SVR", "ORR", "SCR")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation(directory: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    paths = sorted(directory.glob("judged_*.jsonl"))
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record["task_id"])
            if task_id in records:
                raise ValueError(f"duplicate task {task_id!r} in {directory}")
            records[task_id] = record
    return {"files": len(paths), "records": records}


def flatten_evaluation(records: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    excluded = 0
    for task_id, record in records.items():
        metadata = record["manifest_metadata"]
        fulfillable = bool(int(record["source_record"]["fulfillable"]))
        for item in record["judged_trajectories"]:
            trajectory = item["trajectory"]
            judgment = item["judgment"]
            if trajectory.get("candidate_role") != "unprivileged_rollout":
                continue
            if not judgment["scorer"]["verified"]:
                excluded += 1
                continue
            rows.append(
                {
                    "task_id": task_id,
                    "pair_group_id": str(metadata["pair_group_id"]),
                    "fulfillable": fulfillable,
                    "safe": bool(judgment["safe"]),
                    "task_success": bool(judgment["task_success"]),
                    "refused": bool(judgment["refused"]),
                }
            )
    return rows, excluded


def measure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {metric: None for metric in METRICS} | {
            "verified_trajectories": 0,
            "tasks": 0,
            "fulfillable_trajectories": 0,
            "unfulfillable_trajectories": 0,
        }
    fulfillable_rows = [row for row in rows if row["fulfillable"]]
    return {
        "SVR": sum(not row["safe"] for row in rows) / len(rows),
        "ORR": (
            sum(row["refused"] for row in fulfillable_rows) / len(fulfillable_rows)
            if fulfillable_rows
            else None
        ),
        "SCR": sum(row["safe"] and row["task_success"] for row in rows) / len(rows),
        "verified_trajectories": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "fulfillable_trajectories": len(fulfillable_rows),
        "unfulfillable_trajectories": len(rows) - len(fulfillable_rows),
    }


def paired_deltas(
    baseline_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    base_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    adapter_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        base_by_group[row["pair_group_id"]].append(row)
    for row in adapter_rows:
        adapter_by_group[row["pair_group_id"]].append(row)
    shared_groups = sorted(set(base_by_group) & set(adapter_by_group))
    if not shared_groups:
        return {"paired_groups": 0, "deltas": {metric: None for metric in METRICS}}

    base_rows = [row for group in shared_groups for row in base_by_group[group]]
    adapted_rows = [row for group in shared_groups for row in adapter_by_group[group]]
    generator = random.Random(seed)
    draws = {metric: [] for metric in METRICS}
    for sample_index in range(samples):
        sampled_groups = [
            shared_groups[generator.randrange(len(shared_groups))]
            for group_index in shared_groups
        ]
        draw_base = [row for group in sampled_groups for row in base_by_group[group]]
        draw_adapter = [row for group in sampled_groups for row in adapter_by_group[group]]
        base_metrics = measure(draw_base)
        adapter_metrics = measure(draw_adapter)
        for metric in METRICS:
            if base_metrics[metric] is not None and adapter_metrics[metric] is not None:
                draws[metric].append(adapter_metrics[metric] - base_metrics[metric])

    base_metrics = measure(base_rows)
    adapter_metrics = measure(adapted_rows)
    deltas = {}
    for metric, values in draws.items():
        values.sort()
        deltas[metric] = {
            "adapter_minus_base": adapter_metrics[metric] - base_metrics[metric],
            "task_group_bootstrap_95_ci": [
                values[int(0.025 * (len(values) - 1))],
                values[int(0.975 * (len(values) - 1))],
            ],
        }
    return {"paired_groups": len(shared_groups), "deltas": deltas}


def percentage(value: float | None) -> str:
    return "pending" if value is None else f"{100 * value:.1f}%"


def markdown_report(report: dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    baseline = evaluation["baseline"]
    adapter = evaluation["adapter"]
    production = report["production"]
    lines = [
        "# Production Experiment Results",
        "",
        f"**Status:** {report['status']}",
        "",
        "This report is generated from raw execution shards. It excludes unverified judge outputs and does not substitute task-level similarity for same-state trajectory pairs.",
        "",
        "## Held-out Agent-SafetyBench evaluation",
        "",
        "| Model | SVR ↓ | ORR ↓ | SCR ↑ | Verified trajectories | Tasks |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Qwen3-4B base | {percentage(baseline['SVR'])} | {percentage(baseline['ORR'])} | {percentage(baseline['SCR'])} | {baseline['verified_trajectories']} | {baseline['tasks']} |",
        f"| HFDTree full adapter | {percentage(adapter['SVR'])} | {percentage(adapter['ORR'])} | {percentage(adapter['SCR'])} | {adapter['verified_trajectories']} | {adapter['tasks']} |",
        "",
        f"The base evaluation has {evaluation['baseline_files']} judged shard files; {evaluation['baseline_unverified_excluded']} unverified unprivileged trajectories were excluded. The adapter has {evaluation['adapter_files']} judged shard files; {evaluation['adapter_unverified_excluded']} unverified trajectories were excluded.",
        "",
    ]
    paired = evaluation["paired"]
    if paired is None:
        lines.extend(
            [
                "Adapter rollouts/judgments are not complete, so no method comparison or confidence interval is reported.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Paired comparison uses {paired['paired_groups']} held-out task groups and task-group bootstrap resampling.",
                "",
                "| Metric | Adapter − base | 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for metric in METRICS:
            delta = paired["deltas"][metric]
            lower, upper = delta["task_group_bootstrap_95_ci"]
            lines.append(
                f"| {metric} | {100 * delta['adapter_minus_base']:+.1f} pp | [{100 * lower:+.1f}, {100 * upper:+.1f}] pp |"
            )
        lines.append("")

    lines.extend(
        [
            "## Data validity and cost",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| Development training tasks collected | {production['collected_tasks']} |",
            f"| Verified same-state pairs | {production['verified_pairs']} ({100 * production['pair_eligibility_rate']:.1f}% eligibility) |",
            f"| Environment executions for collection | {production['environment_executions']} |",
            f"| Reference-environment executions | {production['reference_environment_executions']} |",
            f"| Scored training nodes | {production['scored_training_nodes']} |",
            f"| Fixed-token scoring calls | {production['scoring_cost']['forward_calls']} |",
            f"| Fixed-token scoring wall time | {production['scoring_cost']['wall_seconds_sum']:.1f} s |",
            "",
            "## Matched training controls",
            "",
            "All listed variants use the same optimizer configuration and one seed (`42`). These are cost records, not held-out performance comparisons.",
            "",
            "| Variant | Examples | Fallback examples | Steps | GPU hours | Energy (Wh) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in report["training"]["variants"]:
        lines.append(
            f"| {variant['control']} | {variant['examples']} | {variant['fallback_examples']} | {variant['optimizer_steps']} | {variant['gpu_hours']:.3f} | {variant['gpu_energy_wh']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "bash scripts/finalize_qwen_production_training.sh",
            "python scripts/render_production_results.py \\",
            "  --production-dir artifacts/production/qwen3_4b_g8_v2 \\",
            "  --output results/production_experiment_results.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")

    production_dir = args.production_dir
    production = read_json(production_dir / "merged" / "production_summary.json")
    training = read_json(production_dir / "training" / "variant_summary.json")
    baseline_evaluation = load_evaluation(production_dir / "holdout_eval" / "base" / "judged")
    adapter_evaluation = load_evaluation(production_dir / "holdout_eval" / "adapter" / "judged")
    baseline_rows, baseline_excluded = flatten_evaluation(baseline_evaluation["records"])
    adapter_rows, adapter_excluded = flatten_evaluation(adapter_evaluation["records"])
    adapter_complete = adapter_evaluation["files"] == baseline_evaluation["files"] and bool(adapter_rows)
    paired = (
        paired_deltas(baseline_rows, adapter_rows, args.bootstrap_samples, args.seed)
        if adapter_complete
        else None
    )
    report = {
        "status": "complete" if adapter_complete else "incomplete: adapter holdout evaluation pending",
        "production": production,
        "training": training,
        "evaluation": {
            "baseline": measure(baseline_rows),
            "adapter": measure(adapter_rows),
            "baseline_files": baseline_evaluation["files"],
            "adapter_files": adapter_evaluation["files"],
            "baseline_unverified_excluded": baseline_excluded,
            "adapter_unverified_excluded": adapter_excluded,
            "paired": paired,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown_report(report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
