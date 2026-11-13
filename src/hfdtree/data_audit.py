"""Audits for the official task-pair artifact used by the paper experiments.

The artifact contains *task records*, not executed trajectories.  This module
keeps that distinction explicit while checking source fidelity, pair validity,
and the cardinality of the Agent-SafetyBench matching graph.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


Record = Mapping[str, object]
PairPredicate = Callable[[Record, Record], bool]


@dataclass(frozen=True, slots=True)
class MatchingResult:
    """A maximum-cardinality bipartite matching."""

    cardinality: int
    pairs: tuple[tuple[int, int], ...]


def load_json(path: str | Path) -> object:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def risks(record: Record) -> frozenset[str]:
    return frozenset(str(value) for value in record.get("risks", []))


def environments(record: Record) -> frozenset[str]:
    return frozenset(
        str(item.get("name", ""))
        for item in record.get("environments", [])  # type: ignore[union-attr]
        if isinstance(item, Mapping) and item.get("name")
    )


def tools(record: Record) -> frozenset[str]:
    values: set[str] = set()
    for environment in record.get("environments", []):  # type: ignore[union-attr]
        if not isinstance(environment, Mapping):
            continue
        values.update(str(tool) for tool in environment.get("tools", []))
    return frozenset(values)


PAIRING_CONDITIONS: dict[str, PairPredicate] = {
    "same_risk_environment_shared_tool": lambda positive, negative: bool(
        risks(positive) & risks(negative)
        and environments(positive) & environments(negative)
        and tools(positive) & tools(negative)
    ),
    "same_environment_shared_tool": lambda positive, negative: bool(
        environments(positive) & environments(negative) and tools(positive) & tools(negative)
    ),
    "shared_tool": lambda positive, negative: bool(tools(positive) & tools(negative)),
    "shared_environment": lambda positive, negative: bool(
        environments(positive) & environments(negative)
    ),
    "shared_risk": lambda positive, negative: bool(risks(positive) & risks(negative)),
}


def maximum_bipartite_matching(
    positives: Sequence[Record], negatives: Sequence[Record], predicate: PairPredicate
) -> MatchingResult:
    """Return a deterministic Hopcroft-Karp maximum-cardinality matching."""

    adjacency = [
        tuple(index for index, negative in enumerate(negatives) if predicate(positive, negative))
        for positive in positives
    ]
    pair_left = [-1] * len(positives)
    pair_right = [-1] * len(negatives)
    distance = [0] * len(positives)
    infinity = len(positives) + len(negatives) + 1

    def breadth_first() -> bool:
        queue: deque[int] = deque()
        found = False
        for left in range(len(positives)):
            if pair_left[left] == -1:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = infinity
        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                matched_left = pair_right[right]
                if matched_left == -1:
                    found = True
                elif distance[matched_left] == infinity:
                    distance[matched_left] = distance[left] + 1
                    queue.append(matched_left)
        return found

    def depth_first(left: int) -> bool:
        for right in adjacency[left]:
            matched_left = pair_right[right]
            if matched_left == -1 or (
                distance[matched_left] == distance[left] + 1 and depth_first(matched_left)
            ):
                pair_left[left] = right
                pair_right[right] = left
                return True
        distance[left] = infinity
        return False

    while breadth_first():
        for left in range(len(positives)):
            if pair_left[left] == -1:
                depth_first(left)

    pairs = tuple((left, right) for left, right in enumerate(pair_left) if right != -1)
    return MatchingResult(cardinality=len(pairs), pairs=pairs)


def matching_sensitivity(records: Sequence[Record]) -> dict[str, int]:
    positives = [record for record in records if int(record.get("fulfillable", -1)) == 1]
    negatives = [record for record in records if int(record.get("fulfillable", -1)) == 0]
    return {
        name: maximum_bipartite_matching(positives, negatives, predicate).cardinality
        for name, predicate in PAIRING_CONDITIONS.items()
    }


def _source_index(records: Iterable[Record]) -> dict[object, Record]:
    index: dict[object, Record] = {}
    for record in records:
        identifier = record.get("id")
        if identifier in index:
            raise ValueError(f"duplicate source id: {identifier}")
        index[identifier] = record
    return index


def _agentharm_rows(path: str | Path) -> list[Record]:
    value = load_json(path)
    if not isinstance(value, Mapping) or not isinstance(value.get("behaviors"), list):
        raise ValueError(f"unexpected AgentHarm source schema: {path}")
    return value["behaviors"]  # type: ignore[return-value]


def audit_pair_artifact(
    artifact_path: str | Path,
    *,
    agent_safety_source: str | Path,
    agentharm_source_dir: str | Path,
) -> dict[str, object]:
    """Audit the pair artifact against pinned official source checkouts."""

    artifact = load_json(artifact_path)
    source_records = load_json(agent_safety_source)
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("datasets"), Mapping):
        raise ValueError("pair artifact must contain a datasets object")
    if not isinstance(source_records, list):
        raise ValueError("Agent-SafetyBench source must be a JSON list")

    artifact_agentharm = artifact["datasets"].get("AgentHarm", [])  # type: ignore[union-attr]
    artifact_safety = artifact["datasets"].get("AgentSafetyBench", [])  # type: ignore[union-attr]
    if not isinstance(artifact_agentharm, list) or not isinstance(artifact_safety, list):
        raise ValueError("artifact datasets must be arrays")

    official_safety_index = _source_index(source_records)
    safety_positive_ids: list[object] = []
    safety_negative_ids: list[object] = []
    invalid_strict_pairs: list[str] = []
    inexact_safety_records: list[str] = []
    formula_inconsistencies: list[str] = []
    for pair in artifact_safety:
        positive = pair["positive_sample"]
        negative = pair["negative_sample"]
        pair_id = str(pair.get("pair_id"))
        safety_positive_ids.append(positive.get("id"))
        safety_negative_ids.append(negative.get("id"))
        if official_safety_index.get(positive.get("id")) != positive:
            inexact_safety_records.append(f"{pair_id}:positive")
        if official_safety_index.get(negative.get("id")) != negative:
            inexact_safety_records.append(f"{pair_id}:negative")
        if not PAIRING_CONDITIONS["same_risk_environment_shared_tool"](positive, negative):
            invalid_strict_pairs.append(pair_id)
        components = pair.get("pairing_method", {}).get("components", {})
        if isinstance(components, Mapping):
            expected = (
                0.35 * float(components.get("risk_jaccard", 0.0))
                + 0.30 * float(components.get("environment_jaccard", 0.0))
                + 0.20 * float(components.get("tool_jaccard", 0.0))
                + 0.05 * float(components.get("failure_mode_jaccard", 0.0))
                + 0.10 * float(components.get("instruction_token_jaccard", 0.0))
            )
            if abs(expected - float(components.get("score", 0.0))) > 5e-6:
                formula_inconsistencies.append(pair_id)

    source_directory = Path(agentharm_source_dir)
    agentharm_sources: dict[tuple[str, str], Record] = {}
    agentharm_source_hashes: dict[str, str] = {}
    for split in ("test_public", "validation"):
        for polarity in ("benign", "harmful"):
            path = source_directory / f"{polarity}_behaviors_{split}.json"
            agentharm_source_hashes[path.name] = sha256_file(path)
            for record in _agentharm_rows(path):
                agentharm_sources[(split, polarity, str(record.get("id")))] = record

    inexact_agentharm_records: list[str] = []
    same_target_functions = 0
    split_counts: Counter[str] = Counter()
    for pair in artifact_agentharm:
        pair_id = str(pair.get("pair_id"))
        split = str(pair.get("source_split"))
        positive = pair["positive_sample"]
        negative = pair["negative_sample"]
        split_counts[split] += 1
        if agentharm_sources.get((split, "benign", str(positive.get("id")))) != positive:
            inexact_agentharm_records.append(f"{pair_id}:positive")
        if agentharm_sources.get((split, "harmful", str(negative.get("id")))) != negative:
            inexact_agentharm_records.append(f"{pair_id}:negative")
        if positive.get("id") != negative.get("id"):
            inexact_agentharm_records.append(f"{pair_id}:id_mismatch")
        if positive.get("target_functions") == negative.get("target_functions"):
            same_target_functions += 1

    sensitivity = matching_sensitivity(source_records)
    strict_maximum = sensitivity["same_risk_environment_shared_tool"]
    current_is_matching = (
        len(set(safety_positive_ids)) == len(safety_positive_ids)
        and len(set(safety_negative_ids)) == len(safety_negative_ids)
        and not invalid_strict_pairs
    )
    current_is_maximum = current_is_matching and len(artifact_safety) == strict_maximum

    safety_risks = sorted(
        {
            value
            for pair in artifact_safety
            for value in risks(pair["positive_sample"]) & risks(pair["negative_sample"])
        }
    )
    safety_environments = sorted(
        {
            value
            for pair in artifact_safety
            for value in environments(pair["positive_sample"])
            & environments(pair["negative_sample"])
        }
    )
    safety_tools = sorted(
        {
            value
            for pair in artifact_safety
            for value in tools(pair["positive_sample"]) & tools(pair["negative_sample"])
        }
    )

    return {
        "provenance": {
            "artifact_sha256": sha256_file(artifact_path),
            "agent_safety_bench_source_sha256": sha256_file(agent_safety_source),
            "agentharm_source_sha256": dict(sorted(agentharm_source_hashes.items())),
        },
        "artifact": {
            "declared_counts": artifact.get("counts"),
            "observed_counts": {
                "AgentHarm": len(artifact_agentharm),
                "AgentSafetyBench": len(artifact_safety),
                "total_pairs": len(artifact_agentharm) + len(artifact_safety),
            },
        },
        "source_counts": {
            "AgentSafetyBench": len(source_records),
            "AgentSafetyBench_fulfillable_1": sum(
                int(record.get("fulfillable", -1)) == 1 for record in source_records
            ),
            "AgentSafetyBench_fulfillable_0": sum(
                int(record.get("fulfillable", -1)) == 0 for record in source_records
            ),
            "AgentHarm_pairs_by_split": dict(sorted(split_counts.items())),
        },
        "source_fidelity": {
            "agentharm_exact": not inexact_agentharm_records,
            "agentharm_mismatches": inexact_agentharm_records,
            "agent_safety_bench_exact": not inexact_safety_records,
            "agent_safety_bench_mismatches": inexact_safety_records,
        },
        "agentharm": {
            "native_id_pairs": len(artifact_agentharm) - sum(
                item.endswith(":id_mismatch") for item in inexact_agentharm_records
            ),
            "same_target_functions_pairs": same_target_functions,
            "all_pairs": len(artifact_agentharm),
            "training_allowed": False,
            "reason": "The official dataset card requests evaluation-only use.",
        },
        "agent_safety_bench": {
            "strict_pairs_valid": not invalid_strict_pairs,
            "invalid_strict_pairs": invalid_strict_pairs,
            "unique_positive_records": len(set(safety_positive_ids)),
            "unique_negative_records": len(set(safety_negative_ids)),
            "score_formula_consistent": not formula_inconsistencies,
            "score_formula_inconsistencies": formula_inconsistencies,
            "maximum_matching_by_condition": sensitivity,
            "current_pairing_is_maximum_cardinality_under_strict_condition": current_is_maximum,
            "coverage": {
                "shared_risks": len(safety_risks),
                "shared_environments": len(safety_environments),
                "shared_tools": len(safety_tools),
            },
        },
        "semantic_validity": {
            "artifact_level": "official_task_or_record_pairs",
            "verified_executed_trajectory_pairs": 0,
            "usable_as_hfdtree_c_plus_c_minus_without_execution": False,
        },
    }


def audit_markdown(report: Mapping[str, object]) -> str:
    """Render the essential audit findings without exposing benchmark prompts."""

    artifact = report["artifact"]
    fidelity = report["source_fidelity"]
    agentharm = report["agentharm"]
    safety = report["agent_safety_bench"]
    sensitivity = safety["maximum_matching_by_condition"]
    return "\n".join(
        [
            "# Official pair-data audit",
            "",
            "## Reproduced results",
            "",
            f"- Observed pairs: `{artifact['observed_counts']}`",
            f"- Embedded AgentHarm records exactly match source: `{fidelity['agentharm_exact']}`",
            "- Embedded Agent-SafetyBench records exactly match source: "
            f"`{fidelity['agent_safety_bench_exact']}`",
            f"- AgentHarm pairs with identical target functions: "
            f"`{agentharm['same_target_functions_pairs']}/{agentharm['all_pairs']}`",
            f"- Current strict Agent-SafetyBench pairing is maximum-cardinality: "
            f"`{safety['current_pairing_is_maximum_cardinality_under_strict_condition']}`",
            "",
            "## Pairing-condition sensitivity",
            "",
            "| Condition | Maximum one-to-one pairs |",
            "|---|---:|",
            f"| Same risk + same environment + shared tool | "
            f"{sensitivity['same_risk_environment_shared_tool']} |",
            f"| Same environment + shared tool | {sensitivity['same_environment_shared_tool']} |",
            f"| Shared tool | {sensitivity['shared_tool']} |",
            f"| Shared environment | {sensitivity['shared_environment']} |",
            f"| Shared risk | {sensitivity['shared_risk']} |",
            "",
            "## Interpretation boundary",
            "",
            "The 337 objects are official task/record pairs. They contain zero verified executed "
            "trajectory pairs and cannot be used as `c+`/`c-` until both outcomes are produced and "
            "scored from a compatible environment state. AgentHarm must remain evaluation-only "
            "because its official dataset card asks researchers not to train on its examples.",
            "",
        ]
    )
