import pytest

from hfdtree.agentharm import records_from_rows, summarize_agentharm


def test_agentharm_schema_and_privacy_preserving_summary() -> None:
    records = records_from_rows(
        [
            {"id": "1", "prompt": "harmful prompt", "category": "Fraud", "split": "test"},
            {"id": "2", "prompt": "another prompt", "category": "Hate", "split": "test"},
        ]
    )
    assert summarize_agentharm(records) == {
        "records": 2,
        "categories": {"Fraud": 1, "Hate": 1},
        "splits": {"test": 2},
        "unique_ids": 2,
    }


def test_agentharm_rejects_duplicates_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        records_from_rows(
            [
                {"id": "1", "prompt": "x", "category": "Fraud"},
                {"id": "1", "prompt": "y", "category": "Hate"},
            ]
        )
    with pytest.raises(ValueError, match="missing fields"):
        records_from_rows([{"id": "1", "prompt": "x"}])
