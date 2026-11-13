#!/usr/bin/env python3
"""Verify matched budgets and summarize completed production variant runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VARIANTS = (
    "full",
    "no_fallback",
    "propagation_without_centering",
    "two_sided_local_only",
    "two_sided_local_with_fallback",
    "one_sided_tree",
    "outcome_tree_equal_information",
    "direct_bireference_equal_information",
)

MATCHED_FALLBACK_VARIANTS = (
    "full",
    "no_fallback",
    "propagation_without_centering_matched_fallback",
    "two_sided_local_matched_fallback",
    "one_sided_tree_matched_fallback",
    "outcome_tree_matched_fallback",
    "direct_bireference_matched_fallback",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    budgets = set()
    variants = MATCHED_FALLBACK_VARIANTS if args.matched_fallback else VARIANTS
    for variant in variants:
        path = args.training_root / f"{variant}_seed{args.seed}" / "training_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        hyper = metrics["hyperparameters"]
        budgets.add(
            tuple(
                (name, hyper[name])
                for name in (
                    "learning_rate",
                    "weight_decay",
                    "batch_size",
                    "gradient_accumulation_steps",
                    "max_steps",
                    "max_length",
                    "lora_rank",
                    "lora_alpha",
                    "delta",
                    "epsilon",
                )
            )
        )
        records.append(
            {
                "control": variant,
                "examples": metrics["examples"],
                "fallback_examples": metrics["fallback_examples"],
                "optimizer_steps": metrics["optimizer_steps"],
                "tokens_processed": metrics["tokens_processed"],
                "wall_seconds": metrics["wall_seconds"],
                "gpu_hours": metrics["gpu_hours"],
                "gpu_energy_wh": metrics["gpu_energy_wh"],
                "peak_gpu_memory_bytes": metrics["peak_gpu_memory_bytes"],
                "mean_recent_loss": metrics["mean_recent_loss"],
                "hyperparameters": hyper,
            }
        )
    if len(budgets) != 1:
        raise ValueError("training variants do not share an identical optimizer budget")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"matched_budget": True, "seed": args.seed, "variants": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    fields = [key for key in records[0] if key != "hyperparameters"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    print(
        json.dumps(
            {"matched_budget": True, "variants": len(records), "summary": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
