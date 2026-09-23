import math

import pytest

from hfdtree.evidence import compute_evidence, soft_threshold


def test_two_sided_evidence_and_threshold() -> None:
    result = compute_evidence([-0.1, -0.2], [-0.4, -0.5], temperature=0.5, delta=0.05)
    assert result.token_sum == pytest.approx(0.6)
    assert result.normalized == pytest.approx(0.3)
    assert result.bounded == pytest.approx(math.tanh(0.25))


def test_dead_zone_and_sign() -> None:
    assert soft_threshold(0.05, delta=0.05) == 0.0
    assert soft_threshold(-0.04, delta=0.05) == 0.0
    assert soft_threshold(-0.2, temperature=0.5, delta=0.05) < 0.0


def test_evidence_rejects_misaligned_tokens() -> None:
    with pytest.raises(ValueError, match="identical action tokens"):
        compute_evidence([-0.1], [-0.2, -0.3])
