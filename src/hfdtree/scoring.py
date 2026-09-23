"""Optional Hugging Face helpers for scoring fixed action token IDs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def score_action_tokens(
    model: object,
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
    action_mask: "torch.Tensor",
) -> list[list[float]]:
    """Score fixed action tokens, excluding prompts and padding.

    Build positive- and negative-context batches separately but keep the same
    action token IDs and action_mask. This avoids retokenization ambiguity at
    the prompt/action boundary.
    """

    import torch
    import torch.nn.functional as F

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have shape [B,T]")
    if action_mask.shape != input_ids.shape:
        raise ValueError("action_mask must match input_ids")
    if torch.any(action_mask[:, 0]):
        raise ValueError("the first token cannot be scored by a causal LM")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logprobs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        target = input_ids[:, 1:]
        selected = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        selected_mask = action_mask[:, 1:].bool() & attention_mask[:, 1:].bool()

    rows: list[list[float]] = []
    for values, mask in zip(selected, selected_mask, strict=True):
        if not torch.any(mask):
            raise ValueError("each row needs at least one action token")
        rows.append(values[mask].detach().cpu().tolist())
    return rows
