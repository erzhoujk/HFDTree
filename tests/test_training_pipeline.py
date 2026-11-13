from hfdtree.evidence_pipeline import score_task_rollouts
from hfdtree.training import prepare_examples


class FakeScorer:
    def score(self, messages, tools, action_token_ids):
        system = str(messages[0].get("content", ""))
        if "safe/successful" in system:
            value = -0.1
        elif "violating" in system:
            value = -1.1
        else:
            value = -0.5
        return [value] * len(action_token_ids)


class FakeTokenizer:
    name_or_path = "fake"

    def apply_chat_template(self, messages, **kwargs):
        return [10, 11]


def _turn(action, observation, state, token):
    return {
        "action": action,
        "observation": observation,
        "raw_text": str(action),
        "action_token_ids": [token],
        "state_fingerprint_before": state,
        "state_fingerprint_after": state + "-after",
        "prefix_messages": [{"role": "system", "content": "base"}],
    }


def test_scoring_emits_controlled_credit_variants() -> None:
    root = _turn({"kind": "tool", "tool_name": "a"}, {"ok": 1}, "root", 3)
    left = _turn({"kind": "content", "content": "safe"}, {"terminal": True}, "left", 4)
    right = _turn({"kind": "content", "content": "bad"}, {"terminal": True}, "right", 5)
    collection = {
        "task_id": "task-1",
        "source_record_id": 1,
        "rollouts": [
            {"turns": [root, left]},
            {"turns": [root, right]},
        ],
    }
    pair = {
        "positive_trajectory": {"turns": [root, left]},
        "negative_trajectory": {"turns": [root, right]},
    }
    rows = score_task_rollouts(FakeScorer(), collection, pair, [])
    assert len(rows) == 3
    assert {row["depth"] for row in rows} == {1, 2}
    assert all("credit_full" in row and "credit_one_sided_tree" in row for row in rows)
    assert rows[0]["reference_valid"] is True


def test_prepared_training_examples_select_state_valid_fallback() -> None:
    row = {
        "credit_full": 0.01,
        "credit_propagation_without_centering": 0.01,
        "credit_two_sided_local_only": 0.01,
        "credit_one_sided_tree": 0.01,
        "normalized_evidence": 0.01,
        "reference_valid": True,
        "reference_action_token_ids": [8, 9],
        "action_token_ids": [3],
        "tools": [],
        "prefix_messages": [{"role": "system", "content": "base"}],
    }
    examples = prepare_examples([row], FakeTokenizer(), variant="full")
    assert len(examples) == 1
    assert examples[0].fallback is True
    assert examples[0].input_ids[-2:] == (8, 9)
    assert examples[0].weight == 1.0


def test_matched_fallback_variants_use_full_objective_selection() -> None:
    row = {
        "credit_full": 0.01,
        "credit_two_sided_local_only": 0.8,
        "normalized_evidence": 0.01,
        "reference_valid": True,
        "reference_action_token_ids": [8, 9],
        "action_token_ids": [3],
        "tools": [],
        "prefix_messages": [{"role": "system", "content": "base"}],
    }
    matched = prepare_examples(
        [row], FakeTokenizer(), variant="two_sided_local_matched_fallback"
    )
    legacy = prepare_examples(
        [row], FakeTokenizer(), variant="two_sided_local_with_fallback"
    )
    assert matched[0].fallback is True
    assert matched[0].input_ids[-2:] == (8, 9)
    assert legacy[0].fallback is False
    assert legacy[0].input_ids[-1:] == (3,)
