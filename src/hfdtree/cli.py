"""Command-line interface for propagating pre-scored rollout trees."""

from __future__ import annotations

import argparse

from .io import load_jsonl, write_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute HFDTree turn-level credits")
    parser.add_argument("input", help="rollout JSONL")
    parser.add_argument("-o", "--output", default="credits.jsonl")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="fail fast on malformed/empty task trees before propagation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trees = load_jsonl(args.input, temperature=args.temperature, delta=args.delta)
    if not trees:
        raise ValueError("input contains no rollout records")
    for tree in trees.values():
        if args.validate and not tree.root.children:
            raise ValueError(f"task {tree.task_id!r} has no turns")
        tree.propagate(gamma=args.gamma)
    write_results(trees.values(), args.output)
    print(f"wrote {sum(len(tree.results()) for tree in trees.values())} nodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
