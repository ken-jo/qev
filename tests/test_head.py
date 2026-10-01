import pytest
import torch

from veyra.head import DecisionHead, HeadConfig


@pytest.fixture
def head():
    torch.manual_seed(19)
    return DecisionHead(
        HeadConfig(input_size=12, hidden_size=16, layers=2, attention_heads=4)
    ).eval()


def test_permutation_equivariance(head):
    features = torch.randn(2, 7, 12)
    valid = torch.ones(2, 7, dtype=torch.bool)
    types = torch.tensor([0, 2])
    levels = torch.zeros(2, 7)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    original = head(features, valid, types, levels)
    permuted = head(features[:, permutation], valid[:, permutation], types, levels[:, permutation])
    torch.testing.assert_close(permuted, original[:, permutation], atol=1e-6, rtol=1e-5)


def test_padding_does_not_change_valid_scores(head):
    features = torch.randn(1, 3, 12)
    original = head(
        features, torch.ones(1, 3, dtype=torch.bool), torch.tensor([0]), torch.zeros(1, 3)
    )
    padded = torch.cat([features, torch.full((1, 4, 12), float("nan"))], dim=1)
    valid = torch.tensor([[True, True, True, False, False, False, False]])
    result = head(padded, valid, torch.tensor([0]), torch.zeros(1, 7))
    torch.testing.assert_close(original, result[:, :3], atol=1e-6, rtol=1e-5)
    assert torch.isneginf(result[:, 3:]).all()
    assert torch.softmax(result, -1)[:, 3:].sum() == 0


@pytest.mark.parametrize("count", [2, 5, 16])
def test_different_candidate_counts_share_parameters(head, count):
    before = sum(p.numel() for p in head.parameters())
    result = head(
        torch.randn(1, count, 12),
        torch.ones(1, count, dtype=torch.bool),
        torch.tensor([1]),
        torch.linspace(0, 1, count).unsqueeze(0),
    )
    assert result.shape == (1, count)
    assert sum(p.numel() for p in head.parameters()) == before


def test_gradients_reach_head(head):
    features = torch.randn(3, 4, 12)
    logits = head(
        features, torch.ones(3, 4, dtype=torch.bool), torch.tensor([0, 1, 2]), torch.zeros(3, 4)
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    assert head.scorer[-1].weight.grad.abs().sum() > 0
    assert head.project[1].weight.grad.abs().sum() > 0


def test_invalid_input_rejected(head):
    with pytest.raises(ValueError, match="at least two"):
        head(
            torch.randn(1, 2, 12),
            torch.tensor([[True, False]]),
            torch.tensor([0]),
            torch.zeros(1, 2),
        )
    with pytest.raises(ValueError, match="finite"):
        head(
            torch.full((1, 2, 12), float("nan")),
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([0]),
            torch.zeros(1, 2),
        )
