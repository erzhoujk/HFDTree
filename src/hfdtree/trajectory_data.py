"""Construction and validation utilities for execution-grounded training data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def stable_pair_split(
    pair_ids: Sequence[str], *, train_fraction: float = 0.8, seed: str = "hfdtree-v1"
) -> dict[str, str]:
    """Split pair groups deterministically, keeping both records in the same split."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one")
    unique_ids = sorted(set(pair_ids))
    if len(unique_ids) != len(pair_ids):
        raise ValueError("pair ids must be unique")
    ranked = sorted(
        unique_ids,
        key=lambda identifier: hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest(),
    )
    train_count = round(len(ranked) * train_fraction)
    train_ids = set(ranked[:train_count])
    return {
        identifier: "development_train" if identifier in train_ids else "development_holdout"
        for identifier in unique_ids
    }


def build_execution_manifests(
    artifact: Mapping[str, object], *, train_fraction: float = 0.8, seed: str = "hfdtree-v1"
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build honest task-seed manifests, never synthetic trajectories.

    Agent-SafetyBench records may be used for an explicitly in-domain development
    experiment. AgentHarm remains evaluation-only in accordance with its dataset
    card and is represented only by non-prompt indices.
    """

    datasets = artifact.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("artifact must contain a datasets object")
    safety_pairs = datasets.get("AgentSafetyBench")
    agentharm_pairs = datasets.get("AgentHarm")
    if not isinstance(safety_pairs, list) or not isinstance(agentharm_pairs, list):
        raise ValueError("artifact datasets must be arrays")

    split_by_pair = stable_pair_split(
        [str(pair["pair_id"]) for pair in safety_pairs],
        train_fraction=train_fraction,
        seed=seed,
    )
    execution_seeds: list[dict[str, object]] = []
    for pair in safety_pairs:
        pair_id = str(pair["pair_id"])
        split = split_by_pair[pair_id]
        for source_role in ("positive_sample", "negative_sample"):
            record = pair[source_role]
            if not isinstance(record, Mapping):
                raise ValueError(f"{pair_id} contains a non-object source record")
            record_id = record.get("id")
            execution_seeds.append(
                {
                    "schema_version": "1.0",
                    "manifest_type": "environment_rollout_collection_seed",
                    "dataset": "Agent-SafetyBench",
                    "pair_group_id": pair_id,
                    "task_id": f"agent_safety_bench_{record_id}",
                    "split": split,
                    "usage": "in_domain_development_not_official_benchmark_evaluation",
                    "source_role_in_similarity_pair": source_role,
                    "source_record_fulfillable": int(record.get("fulfillable", -1)),
                    "source_record": record,
                    "important_semantics": {
                        "source_role_is_not_trajectory_outcome": True,
                        "fulfillable_is_not_a_safe_trajectory_label": True,
                        "similarity_partner_is_not_a_counterfactual_trajectory": True,
                    },
                    "collection_requirements": {
                        "restore_same_initial_checkpoint_for_each_branch": True,
                        "minimum_rollouts": 8,
                        "record_every_action_and_observation": True,
                        "run_official_environment_scorer": True,
                        "require_safe_success_and_violation_for_pair_eligibility": True,
                        "do_not_synthesize_missing_outcomes": True,
                    },
                }
            )

    evaluation_index: list[dict[str, object]] = []
    for pair in agentharm_pairs:
        positive = pair.get("positive_sample", {})
        negative = pair.get("negative_sample", {})
        evaluation_index.append(
            {
                "schema_version": "1.0",
                "manifest_type": "evaluation_only_pair_index",
                "dataset": "AgentHarm",
                "pair_id": pair.get("pair_id"),
                "source_split": pair.get("source_split"),
                "positive_id": positive.get("id") if isinstance(positive, Mapping) else None,
                "negative_id": negative.get("id") if isinstance(negative, Mapping) else None,
                "category": positive.get("category") if isinstance(positive, Mapping) else None,
                "usage": "evaluation_only_never_training",
                "contains_prompt": False,
            }
        )
    return execution_seeds, evaluation_index


def validate_verified_trajectory_pair(record: Mapping[str, object]) -> None:
    """Reject task-level pseudo-pairs and accept only executed, scored trajectories."""

    if "positive_sample" in record or "negative_sample" in record:
        raise ValueError("task/sample pairs are not executed trajectory pairs")
    required = (
        "task_id",
        "source_record_id",
        "initial_state_fingerprint",
        "positive_trajectory",
        "negative_trajectory",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError("missing trajectory-pair fields: " + ", ".join(missing))
    positive = record["positive_trajectory"]
    negative = record["negative_trajectory"]
    if not isinstance(positive, Mapping) or not isinstance(negative, Mapping):
        raise ValueError("positive_trajectory and negative_trajectory must be objects")
    fingerprint = str(record["initial_state_fingerprint"])
    if not fingerprint:
        raise ValueError("initial_state_fingerprint must be non-empty")
    _validate_executed_trajectory(
        positive, expected_fingerprint=fingerprint, expected_safe=True, require_success=True
    )
    _validate_executed_trajectory(
        negative, expected_fingerprint=fingerprint, expected_safe=False, require_success=False
    )
    if positive.get("trajectory_id") == negative.get("trajectory_id"):
        raise ValueError("positive and negative trajectories must be distinct executions")


def _validate_executed_trajectory(
    trajectory: Mapping[str, object],
    *,
    expected_fingerprint: str,
    expected_safe: bool,
    require_success: bool,
) -> None:
    required = ("trajectory_id", "initial_state_fingerprint", "turns", "outcome", "scorer")
    missing = [field for field in required if field not in trajectory]
    if missing:
        raise ValueError("trajectory is missing fields: " + ", ".join(missing))
    if trajectory.get("initial_state_fingerprint") != expected_fingerprint:
        raise ValueError("trajectory was not executed from the declared common initial state")
    turns = trajectory.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("trajectory turns must be a non-empty array")
    for index, turn in enumerate(turns):
        required_turn_fields = (
            "action",
            "observation",
            "state_fingerprint_before",
            "state_fingerprint_after",
        )
        if not isinstance(turn, Mapping) or any(field not in turn for field in required_turn_fields):
            raise ValueError(
                f"turn {index} must contain action, observation, and before/after state fingerprints"
            )
        before = str(turn["state_fingerprint_before"])
        after = str(turn["state_fingerprint_after"])
        if not before or not after:
            raise ValueError(f"turn {index} state fingerprints must be non-empty")
        if index == 0 and before != expected_fingerprint:
            raise ValueError("trajectory first turn does not match the common initial state")
        if index > 0 and before != str(turns[index - 1]["state_fingerprint_after"]):
            raise ValueError(f"trajectory state chain is discontinuous at turn {index}")
    outcome = trajectory.get("outcome")
    if not isinstance(outcome, Mapping) or outcome.get("safe") is not expected_safe:
        raise ValueError(f"trajectory outcome.safe must be {expected_safe}")
    if require_success and outcome.get("task_success") is not True:
        raise ValueError("positive trajectory must be both safe and task-successful")
    scorer = trajectory.get("scorer")
    if not isinstance(scorer, Mapping):
        raise ValueError("trajectory scorer must be an object")
    if scorer.get("verified") is not True or not scorer.get("name") or not scorer.get("version"):
        raise ValueError("trajectory requires a named, versioned, verified scorer result")


def validate_trajectory_pair_jsonl(path: str | Path) -> dict[str, int]:
    """Validate a file and return counts without printing trajectory content."""

    pairs = 0
    tasks: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError("record must be an object")
                validate_verified_trajectory_pair(record)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid trajectory pair on line {line_number}: {exc}") from exc
            pairs += 1
            tasks.add(str(record["task_id"]))
    if pairs == 0:
        raise ValueError("trajectory-pair file is empty")
    return {"verified_pairs": pairs, "unique_tasks": len(tasks)}


def write_jsonl(records: Sequence[Mapping[str, object]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
