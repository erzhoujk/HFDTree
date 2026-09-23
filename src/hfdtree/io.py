"""JSONL interchange format for pre-scored prefix-branching rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .evidence import compute_evidence
from .tree import HFDTree, Turn


def load_jsonl(
    path: str | Path, *, temperature: float = 0.5, delta: float = 0.05
) -> dict[str, HFDTree]:
    """Load rollout records and build one exact-prefix tree per task.

    Every line is a trajectory with ``task_id`` and ``turns``. A turn supplies
    either ``local_evidence`` (plus optional ``normalized_evidence``) or paired
    ``positive_logprobs`` and ``negative_logprobs`` arrays.
    """

    trees: dict[str, HFDTree] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                task_id = str(record["task_id"])
                turns = tuple(_parse_turn(item, temperature, delta) for item in record["turns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL record on line {line_number}: {exc}") from exc
            trees.setdefault(task_id, HFDTree(task_id=task_id)).add_trajectory(turns)
    return trees


def write_results(trees: Iterable[HFDTree], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for tree in trees:
            for result in tree.results():
                record = json.dumps({"task_id": tree.task_id, **result}, ensure_ascii=False)
                handle.write(record + "\n")


def _parse_turn(item: dict[str, object], temperature: float, delta: float) -> Turn:
    if "local_evidence" in item:
        local = float(item["local_evidence"])
        normalized = item.get("normalized_evidence")
        normalized = float(normalized) if normalized is not None else None
    else:
        evidence = compute_evidence(
            item["positive_logprobs"],  # type: ignore[arg-type]
            item["negative_logprobs"],  # type: ignore[arg-type]
            temperature=temperature,
            delta=delta,
        )
        local = evidence.bounded
        normalized = evidence.normalized
    return Turn(
        action=str(item["action"]),
        observation=str(item.get("observation", "")),
        local_evidence=local,
        normalized_evidence=normalized,
        reference_action=item.get("reference_action"),
        reference_valid=bool(item.get("reference_valid", False)),
    )
