"""PyTorch implementation of HFDTree's per-turn optimization objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class LossOutput:
    loss: "torch.Tensor"
    fallback_mask: "torch.Tensor"
    advantages: "torch.Tensor"


def hfd_loss(
    action_mean_logprobs: "torch.Tensor",
    credits: "torch.Tensor",
    normalized_evidence: "torch.Tensor",
    *,
    reference_mean_logprobs: "torch.Tensor | None" = None,
    reference_valid: "torch.Tensor | None" = None,
    delta: float = 0.05,
    epsilon: float = 0.05,
) -> LossOutput:
    """Equation (6), including state-valid safe-reference fallback.

    All tensors are one-dimensional and aligned by training node. Reference
    entries are only selected when both evidence and propagated credit are weak.
    """

    import torch

    tensors = (action_mean_logprobs, credits, normalized_evidence)
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("action scores, credits, and evidence must be 1-D")
    if not all(t.shape == action_mean_logprobs.shape for t in tensors):
        raise ValueError("action scores, credits, and evidence must have equal shape")
    if delta < 0 or epsilon < 0:
        raise ValueError("delta and epsilon must be non-negative")

    if reference_mean_logprobs is None:
        reference_mean_logprobs = torch.zeros_like(action_mean_logprobs)
    if reference_valid is None:
        reference_valid = torch.zeros_like(action_mean_logprobs, dtype=torch.bool)
    if reference_mean_logprobs.shape != action_mean_logprobs.shape:
        raise ValueError("reference scores must match action score shape")
    if reference_valid.shape != action_mean_logprobs.shape:
        raise ValueError("reference_valid must match action score shape")

    fallback = (
        reference_valid.bool()
        & (normalized_evidence.abs() <= delta)
        & (credits.abs() <= epsilon)
    )
    advantages = credits.detach().clamp(-1.0, 1.0)
    credit_loss = -advantages * action_mean_logprobs
    reference_loss = -reference_mean_logprobs
    per_node = torch.where(fallback, reference_loss, credit_loss)
    return LossOutput(loss=per_node.mean(), fallback_mask=fallback, advantages=advantages)


def mean_action_logprobs(
    logits: "torch.Tensor", labels: "torch.Tensor", *, ignore_index: int = -100
) -> "torch.Tensor":
    """Mean causal-LM log-probability over labeled action tokens per example."""

    import torch
    import torch.nn.functional as F

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(ignore_index)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logprobs = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    counts = mask.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("each example needs at least one labeled action token")
    return (token_logprobs * mask).sum(dim=-1) / counts
