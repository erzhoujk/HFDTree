torch = __import__("pytest").importorskip("torch")

from hfdtree.objective import hfd_loss, mean_action_logprobs


def test_fallback_selection() -> None:
    actions = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    credits = torch.tensor([0.5, 0.01, -0.5])
    evidence = torch.tensor([0.3, 0.01, -0.3])
    references = torch.tensor([-0.2, -0.4, -0.6], requires_grad=True)
    valid = torch.tensor([True, True, False])
    out = hfd_loss(
        actions,
        credits,
        evidence,
        reference_mean_logprobs=references,
        reference_valid=valid,
    )
    assert out.fallback_mask.tolist() == [False, True, False]
    assert out.loss.item() == __import__("pytest").approx((0.5 + 0.4 - 1.5) / 3)


def test_mean_action_logprobs_masks_prompt() -> None:
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2]])
    result = mean_action_logprobs(logits, labels)
    assert result.item() == __import__("pytest").approx(-torch.log(torch.tensor(3.0)).item())
