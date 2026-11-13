#!/usr/bin/env python3
"""Train one reviewer-facing controlled HFDTree variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hfdtree.training import CREDIT_FIELDS, train_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(CREDIT_FIELDS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = train_lora(
        rows,
        model_path=args.model,
        output_dir=args.output_dir,
        variant=args.variant,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        delta=args.delta,
        epsilon=args.epsilon,
        resume=args.resume,
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
