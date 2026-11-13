#!/usr/bin/env python3
"""Add the full method's exact fallback examples to direct bi-reference data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def is_full_fallback(row: dict[str, object], *, delta: float, epsilon: float) -> bool:
    return bool(
        row.get("reference_valid")
        and row.get("reference_action_token_ids")
        and abs(float(row["normalized_evidence"])) <= delta
        and abs(float(row["credit_full"])) <= epsilon
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-bireference", type=Path, required=True)
    parser.add_argument("--scored-nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()

    direct = read_jsonl(args.direct_bireference)
    fallback = [
        row
        for row in read_jsonl(args.scored_nodes)
        if is_full_fallback(row, delta=args.delta, epsilon=args.epsilon)
    ]
    if not fallback:
        raise ValueError("no full-objective fallback examples were selected")

    output = direct + fallback
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "direct_bireference_examples": len(direct),
                "matched_fallback_examples": len(fallback),
                "total_examples": len(output),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
