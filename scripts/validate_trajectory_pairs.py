#!/usr/bin/env python3
"""Fail closed unless every c+/c- pair is executed and scorer-verified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hfdtree.trajectory_data import validate_trajectory_pair_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_pairs", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_trajectory_pair_jsonl(args.trajectory_pairs), sort_keys=True))


if __name__ == "__main__":
    main()
