import json

import pytest

from hfdtree.trajectory_data import (
    build_execution_manifests,
    stable_pair_split,
    validate_trajectory_pair_jsonl,
    validate_verified_trajectory_pair,
)


def _trajectory(identifier, *, safe, success):
    return {
        "trajectory_id": identifier,
        "initial_state_fingerprint": "sha256:state",
        "turns": [
            {
                "action": "act",
                "observation": "observed",
                "state_fingerprint_before": "sha256:state",
                "state_fingerprint_after": "sha256:after",
            }
        ],
        "outcome": {"safe": safe, "task_success": success},
        "scorer": {"name": "official-scorer", "version": "1", "verified": True},
    }


def _verified_pair():
    return {
        "task_id": "task-1",
        "source_record_id": 1,
        "initial_state_fingerprint": "sha256:state",
        "positive_trajectory": _trajectory("safe", safe=True, success=True),
        "negative_trajectory": _trajectory("unsafe", safe=False, success=False),
    }


def test_verified_pair_requires_execution_and_scorer_evidence(tmp_path) -> None:
    record = _verified_pair()
    validate_verified_trajectory_pair(record)
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert validate_trajectory_pair_jsonl(path) == {"verified_pairs": 1, "unique_tasks": 1}


def test_task_pair_cannot_masquerade_as_trajectory_pair() -> None:
    with pytest.raises(ValueError, match="not executed"):
        validate_verified_trajectory_pair(
            {"positive_sample": {"id": 1}, "negative_sample": {"id": 2}}
        )


def test_pair_requires_common_initial_state() -> None:
    record = _verified_pair()
    record["negative_trajectory"]["initial_state_fingerprint"] = "sha256:different"
    with pytest.raises(ValueError, match="common initial state"):
        validate_verified_trajectory_pair(record)


def test_pair_rejects_falsely_declared_and_discontinuous_state_traces() -> None:
    record = _verified_pair()
    record["negative_trajectory"]["turns"][0]["state_fingerprint_before"] = "sha256:different"
    with pytest.raises(ValueError, match="first turn"):
        validate_verified_trajectory_pair(record)

    record = _verified_pair()
    record["positive_trajectory"]["turns"].append(
        {
            "action": "next",
            "observation": "done",
            "state_fingerprint_before": "sha256:not-after",
            "state_fingerprint_after": "sha256:done",
        }
    )
    with pytest.raises(ValueError, match="discontinuous"):
        validate_verified_trajectory_pair(record)


def test_stable_split_keeps_exact_requested_pair_count() -> None:
    pair_ids = [f"pair-{index}" for index in range(10)]
    first = stable_pair_split(pair_ids, train_fraction=0.8, seed="fixed")
    second = stable_pair_split(list(reversed(pair_ids)), train_fraction=0.8, seed="fixed")
    assert first == second
    assert list(first.values()).count("development_train") == 8


def test_manifests_keep_agentharm_prompt_free_and_evaluation_only() -> None:
    artifact = {
        "datasets": {
            "AgentSafetyBench": [
                {
                    "pair_id": "asb-1",
                    "positive_sample": {"id": 1, "fulfillable": 1},
                    "negative_sample": {"id": 2, "fulfillable": 0},
                },
                {
                    "pair_id": "asb-2",
                    "positive_sample": {"id": 3, "fulfillable": 1},
                    "negative_sample": {"id": 4, "fulfillable": 0},
                },
            ],
            "AgentHarm": [
                {
                    "pair_id": "ah-1",
                    "source_split": "test_public",
                    "positive_sample": {"id": "1", "prompt": "do not export", "category": "x"},
                    "negative_sample": {"id": "1", "prompt": "do not export", "category": "x"},
                }
            ],
        }
    }
    execution, evaluation = build_execution_manifests(artifact, train_fraction=0.5)
    assert len(execution) == 4
    assert len(evaluation) == 1
    assert evaluation[0]["usage"] == "evaluation_only_never_training"
    assert evaluation[0]["contains_prompt"] is False
    assert "positive_sample" not in evaluation[0]
    assert "negative_sample" not in evaluation[0]
