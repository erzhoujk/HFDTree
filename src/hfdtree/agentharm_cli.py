"""Command-line schema and coverage check for a local AgentHarm export."""

from __future__ import annotations

import argparse
import json

from .agentharm import load_agentharm, summarize_agentharm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an AgentHarm evaluation export")
    parser.add_argument("source", help="AgentHarm JSON/JSONL export or directory")
    parser.add_argument("--split", help="select a split from a split-to-list JSON export")
    args = parser.parse_args(argv)
    records = load_agentharm(args.source, split=args.split)
    print(json.dumps(summarize_agentharm(records), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
