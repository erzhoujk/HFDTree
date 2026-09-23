"""Two-sided privileged likelihood evidence from HFDTree, equations (2) and (16)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Evidence:
    """Evidence for one realized action.

    ``token_sum`` is the sequence log-likelihood ratio, ``normalized`` is the
    per-token value, and ``bounded`` is the sign-preserving soft-thresholded
    signal used as local tree evidence.
    """

    token_sum: float
    normalized: float
    bounded: float
    token_count: int


def soft_threshold(value: float, *, temperature: float = 0.5, delta: float = 0.05) -> float:
    """Apply psi_T,delta(r) = sign(r) tanh((|r|-delta)_+ / (2T))."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if delta < 0:
        raise ValueError("delta must be non-negative")
    magnitude = max(abs(float(value)) - delta, 0.0)
    if magnitude == 0.0:
        return 0.0
    return math.copysign(math.tanh(magnitude / (2.0 * temperature)), value)


def compute_evidence(
    positive_logprobs: Sequence[float] | Iterable[float],
    negative_logprobs: Sequence[float] | Iterable[float],
    *,
    temperature: float = 0.5,
    delta: float = 0.05,
) -> Evidence:
    """Compute two-sided evidence for the same action under c+ and c-.

    Inputs must contain token log-probabilities for exactly the same realized
    action tokens. Padding and prompt tokens must already be excluded.
    """

    positive = tuple(float(x) for x in positive_logprobs)
    negative = tuple(float(x) for x in negative_logprobs)
    if not positive:
        raise ValueError("an action must contain at least one scored token")
    if len(positive) != len(negative):
        raise ValueError("positive and negative scores must cover identical action tokens")
    if not all(math.isfinite(x) for x in positive + negative):
        raise ValueError("token log-probabilities must be finite")

    token_sum = math.fsum(p - n for p, n in zip(positive, negative, strict=True))
    normalized = token_sum / len(positive)
    return Evidence(
        token_sum=token_sum,
        normalized=normalized,
        bounded=soft_threshold(normalized, temperature=temperature, delta=delta),
        token_count=len(positive),
    )
