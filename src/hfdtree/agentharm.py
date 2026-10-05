"""Safe, evaluation-only adapter for the AgentHarm dataset.

AgentHarm contains prompts, not model scores.  This module deliberately only
validates and summarizes records; it never turns prompts into training data.
Scored rollouts should be produced by an external evaluator and then passed to
``hfdtree``'s normal JSONL tree pipeline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping


REQUIRED_FIELDS = ("id", "prompt", "category")


@dataclass(frozen=True, slots=True)
class AgentHarmRecord:
    """Metadata needed to identify one AgentHarm evaluation item.

    The prompt is retained in memory for the caller's evaluation harness, but
    summary functions never print it, reducing accidental exposure in logs.
    """

    id: str
    prompt: str
    category: str
    name: str = ""
    split: str = ""
    id_original: str = ""


def load_agentharm(source: str | Path, *, split: str | None = None) -> list[AgentHarmRecord]:
    """Load AgentHarm records from JSON, JSONL, or a directory of those files.

    A Hugging Face ``datasets`` cache can also be supplied by passing a dataset
    object through :func:`records_from_rows`; keeping that dependency optional
    keeps the core package lightweight.
    """

    path = Path(source)
    if path.is_dir():
        files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
        if not files:
            raise ValueError(f"no JSON/JSONL files found in {path}")
        rows: list[Mapping[str, object]] = []
        for file in files:
            rows.extend(_read_rows(file, split=split))
    else:
        rows = _read_rows(path, split=split)
    return records_from_rows(rows, split=split)


def records_from_rows(
    rows: Iterable[Mapping[str, object]], *, split: str | None = None
) -> list[AgentHarmRecord]:
    """Validate raw dataset rows without emitting their potentially harmful text."""

    records: list[AgentHarmRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"AgentHarm row {index} is missing fields: {', '.join(missing)}")
        identifier = str(row["id"]).strip()
        prompt = str(row["prompt"]).strip()
        category = str(row["category"]).strip()
        if not identifier or not prompt or not category:
            raise ValueError(f"AgentHarm row {index} has an empty id, prompt, or category")
        if identifier in seen:
            raise ValueError(f"duplicate AgentHarm id: {identifier}")
        seen.add(identifier)
        records.append(
            AgentHarmRecord(
                id=identifier,
                prompt=prompt,
                category=category,
                name=str(row.get("name", "")),
                split=split or str(row.get("split", "")),
                id_original=str(row.get("id_original", "")),
            )
        )
    if not records:
        raise ValueError("AgentHarm source contains no records")
    return records


def summarize_agentharm(records: Iterable[AgentHarmRecord]) -> dict[str, object]:
    """Return privacy-preserving coverage statistics for an evaluation split."""

    items = list(records)
    if not items:
        raise ValueError("cannot summarize an empty record collection")
    return {
        "records": len(items),
        "categories": dict(sorted(Counter(item.category for item in items).items())),
        "splits": dict(sorted(Counter(item.split or "unknown" for item in items).items())),
        "unique_ids": len({item.id for item in items}),
    }


def _read_rows(path: Path, *, split: str | None) -> list[Mapping[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix != ".json":
        raise ValueError("AgentHarm adapter accepts .json or .jsonl files")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if split is not None:
            value = value.get(split)
        elif len(value) == 1:
            value = next(iter(value.values()))
        if isinstance(value, list):
            return value
    raise ValueError("JSON must contain a list of AgentHarm rows or split-to-list mapping")
