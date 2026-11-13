#!/usr/bin/env python3
"""Reproduce the official pair-data audit and matching sensitivity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hfdtree.data_audit import audit_markdown, audit_pair_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--agent-safety-source", type=Path, required=True)
    parser.add_argument("--agentharm-source-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_pair_artifact(
        args.artifact,
        agent_safety_source=args.agent_safety_source,
        agentharm_source_dir=args.agentharm_source_dir,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(audit_markdown(report), encoding="utf-8")
    print(json.dumps(report["artifact"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
