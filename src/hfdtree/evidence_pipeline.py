"""Fixed-token privileged-context scoring and controlled credit variants."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Mapping, Sequence

from .evidence import compute_evidence
from .tree import HFDTree, Turn


def reference_summary(trajectory: Mapping[str, object], *, max_characters: int = 12_000) -> str:
    compact = []
    for turn in trajectory.get("turns", []):
        compact.append({"action": turn.get("action"), "observation": turn.get("observation")})
    text = json.dumps(compact, ensure_ascii=False, sort_keys=True)
    if len(text) > max_characters:
        text = text[:max_characters] + "...[truncated]"
    return text


def inject_privileged_reference(
    messages: Sequence[Mapping[str, object]], *, reference: Mapping[str, object], label: str
) -> list[dict[str, object]]:
    result = deepcopy(list(messages))
    context = (
        f"\n\nPrivileged {label} outcome reference for likelihood evaluation only. "
        "Score the next realized action for compatibility with this outcome; do not replace the "
        "current state with the reference state.\n"
        + reference_summary(reference)
    )
    if result and result[0].get("role") == "system":
        result[0]["content"] = str(result[0].get("content", "")) + context
    else:
        result.insert(0, {"role": "system", "content": context.strip()})
    return result


class FixedActionScorer:
    def __init__(self, model_path: str, *, dtype: str = "bfloat16", device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
        ).to(device).eval()
        self.model_path = model_path
        self.scored_prompt_tokens = 0
        self.scored_action_tokens = 0
        self.forward_calls = 0

    def prompt_ids(
        self, messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]]
    ) -> list[int]:
        kwargs = {
            "tools": list(tools),
            "add_generation_prompt": True,
            "tokenize": True,
        }
        if "qwen3" in self.model.config.model_type.lower() or "qwen3" in self.model_path.lower():
            kwargs["enable_thinking"] = False
        values = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        if isinstance(values, Mapping):
            values = values["input_ids"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values]

    def score(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        action_token_ids: Sequence[int],
    ) -> list[float]:
        torch = self.torch
        prompt = self.prompt_ids(messages, tools)
        action = [int(value) for value in action_token_ids]
        if not action:
            raise ValueError("cannot score an empty action")
        input_ids = torch.tensor([prompt + action], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = input_ids[:, 1:]
            selected = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = len(prompt) - 1
        values = selected[0, start : start + len(action)].float().cpu().tolist()
        if len(values) != len(action) or not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite or incomplete fixed-action score")
        self.scored_prompt_tokens += len(prompt)
        self.scored_action_tokens += len(action)
        self.forward_calls += 1
        return values

    def metrics(self) -> dict[str, object]:
        torch = self.torch
        return {
            "model_path": self.model_path,
            "forward_calls": self.forward_calls,
            "scored_prompt_tokens": self.scored_prompt_tokens,
            "scored_action_tokens": self.scored_action_tokens,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }


def score_task_rollouts(
    scorer: FixedActionScorer,
    collection: Mapping[str, object],
    verified_pair: Mapping[str, object],
    tools: Sequence[Mapping[str, object]],
    *,
    outcome_by_trajectory_id: Mapping[str, Mapping[str, object]] | None = None,
    temperature: float = 0.5,
    delta: float = 0.05,
    gamma: float = 0.9,
) -> list[dict[str, object]]:
    positive = verified_pair["positive_trajectory"]
    negative = verified_pair["negative_trajectory"]
    reference_by_state = {
        turn["state_fingerprint_before"]: turn
        for turn in positive.get("turns", [])
    }
    cached: dict[tuple[str, str], dict[str, object]] = {}
    trajectories = []
    metadata_by_path: dict[tuple[tuple[str, str], ...], dict[str, object]] = {}
    for rollout in collection.get("rollouts", []):
        turns = []
        path = []
        for raw_turn in rollout.get("turns", []):
            action_key = json.dumps(raw_turn["action"], ensure_ascii=False, sort_keys=True)
            observation_key = json.dumps(raw_turn["observation"], ensure_ascii=False, sort_keys=True)
            edge_key = (str(raw_turn["state_fingerprint_before"]), action_key)
            if edge_key not in cached:
                action_ids = raw_turn["action_token_ids"]
                prefix = raw_turn["prefix_messages"]
                positive_scores = scorer.score(
                    inject_privileged_reference(prefix, reference=positive, label="safe/successful"),
                    tools,
                    action_ids,
                )
                negative_scores = scorer.score(
                    inject_privileged_reference(prefix, reference=negative, label="violating"),
                    tools,
                    action_ids,
                )
                marginal_scores = scorer.score(prefix, tools, action_ids)
                evidence = compute_evidence(
                    positive_scores,
                    negative_scores,
                    temperature=temperature,
                    delta=delta,
                )
                one_sided = compute_evidence(
                    positive_scores,
                    marginal_scores,
                    temperature=temperature,
                    delta=delta,
                )
                reference_turn = reference_by_state.get(raw_turn["state_fingerprint_before"])
                cached[edge_key] = {
                    "local_evidence": evidence.bounded,
                    "normalized_evidence": evidence.normalized,
                    "one_sided_evidence": one_sided.bounded,
                    "one_sided_normalized": one_sided.normalized,
                    "positive_mean_logprob": sum(positive_scores) / len(positive_scores),
                    "negative_mean_logprob": sum(negative_scores) / len(negative_scores),
                    "marginal_mean_logprob": sum(marginal_scores) / len(marginal_scores),
                    "prefix_messages": deepcopy(prefix),
                    "tools": deepcopy(list(tools)),
                    "action": deepcopy(raw_turn["action"]),
                    "observation": deepcopy(raw_turn["observation"]),
                    "action_token_ids": list(action_ids),
                    "state_fingerprint_before": raw_turn["state_fingerprint_before"],
                    "reference_valid": reference_turn is not None,
                    "reference_action": deepcopy(reference_turn["action"]) if reference_turn else None,
                    "reference_action_token_ids": list(reference_turn.get("action_token_ids", [])) if reference_turn else [],
                }
            item = cached[edge_key]
            turn = Turn(
                action=action_key,
                observation=observation_key,
                local_evidence=float(item["local_evidence"]),
                normalized_evidence=float(item["normalized_evidence"]),
                reference_action=json.dumps(item["reference_action"], sort_keys=True) if item["reference_valid"] else None,
                reference_valid=bool(item["reference_valid"]),
            )
            turns.append(turn)
            path.append((action_key, observation_key))
            metadata_by_path[tuple(path)] = item
        if turns:
            trajectories.append(turns)

    tree = HFDTree(task_id=collection["task_id"])
    for turns in trajectories:
        tree.add_trajectory(turns)
    tree.propagate(gamma=gamma)

    outcome_credit: dict[tuple[tuple[str, str], ...], float] = {}
    if outcome_by_trajectory_id is not None:
        outcome_tree = HFDTree(task_id=collection["task_id"])
        for rollout, turns in zip(collection.get("rollouts", []), trajectories, strict=True):
            judgment = outcome_by_trajectory_id.get(str(rollout.get("trajectory_id")), {})
            reward = (
                1.0
                if judgment.get("safe") is True and judgment.get("task_success") is True
                else -1.0
                if judgment.get("safe") is False
                else 0.0
            )
            outcome_tree.add_trajectory(
                [
                    Turn(
                        action=turn.action,
                        observation=turn.observation,
                        local_evidence=reward if index == len(turns) - 1 else 0.0,
                    )
                    for index, turn in enumerate(turns)
                ]
            )
        outcome_tree.propagate(gamma=gamma)
        outcome_credit = {node.path: node.credit for node in outcome_tree.nodes()}

    one_sided_tree = HFDTree(task_id=collection["task_id"])
    for turns in trajectories:
        one_sided_tree.add_trajectory(
            [
                Turn(
                    action=turn.action,
                    observation=turn.observation,
                    local_evidence=float(metadata_by_path[tuple((node.action, node.observation) for node in turns[: index + 1])]["one_sided_evidence"]),
                )
                for index, turn in enumerate(turns)
            ]
        )
    one_sided_tree.propagate(gamma=gamma)
    one_sided_credit = {node.path: node.credit for node in one_sided_tree.nodes()}

    output = []
    for node in sorted(tree.nodes(), key=lambda value: value.node_id):
        item = deepcopy(metadata_by_path[node.path])
        item.update(
            {
                "task_id": collection["task_id"],
                "source_record_id": collection["source_record_id"],
                "node_id": node.node_id,
                "depth": node.depth,
                "visits": node.visits,
                "transition_probability": node.transition_probability,
                "credit_full": node.credit,
                "credit_propagation_without_centering": node.belief,
                "credit_two_sided_local_only": node.turn.local_evidence,
                "credit_one_sided_tree": one_sided_credit[node.path],
                "credit_outcome_tree": outcome_credit.get(node.path, 0.0),
                "utility": node.utility,
                "belief": node.belief,
            }
        )
        output.append(item)
    return output
