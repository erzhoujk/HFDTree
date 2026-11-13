from hfdtree.data_audit import PAIRING_CONDITIONS, maximum_bipartite_matching


def _record(identifier, *, risk, environment, tool, fulfillable):
    return {
        "id": identifier,
        "risks": [risk],
        "environments": [{"name": environment, "tools": [tool]}],
        "fulfillable": fulfillable,
    }


def test_maximum_matching_finds_augmenting_path() -> None:
    positives = [
        _record(1, risk="r", environment="e", tool="a", fulfillable=1),
        _record(2, risk="r", environment="e", tool="b", fulfillable=1),
    ]
    negatives = [
        {
            "id": 3,
            "risks": ["r"],
            "environments": [{"name": "e", "tools": ["a", "b"]}],
            "fulfillable": 0,
        },
        _record(4, risk="r", environment="e", tool="a", fulfillable=0),
    ]
    result = maximum_bipartite_matching(
        positives,
        negatives,
        PAIRING_CONDITIONS["same_risk_environment_shared_tool"],
    )
    assert result.cardinality == 2
    assert len({right for _, right in result.pairs}) == 2


def test_pairing_conditions_are_strictly_nested_for_example() -> None:
    positive = _record(1, risk="r1", environment="e", tool="t", fulfillable=1)
    negative = _record(2, risk="r2", environment="e", tool="t", fulfillable=0)
    assert not PAIRING_CONDITIONS["same_risk_environment_shared_tool"](
        positive, negative
    )
    assert PAIRING_CONDITIONS["same_environment_shared_tool"](positive, negative)
    assert PAIRING_CONDITIONS["shared_tool"](positive, negative)
    assert PAIRING_CONDITIONS["shared_environment"](positive, negative)
    assert not PAIRING_CONDITIONS["shared_risk"](positive, negative)
