import math

import pytest
import torch

from veyra.features import FeatureSample, collate
from veyra.training import distribution_loss, summarize_predictions


def sample(target, type_id=0):
    count = len(target)
    return FeatureSample(
        features=torch.zeros(count, 12),
        context=torch.zeros(12),
        levels=torch.linspace(0, 1, count),
        targets=torch.tensor(target),
        type_id=type_id,
        record_id="r",
        group_id="g",
        split="dev",
        family="test",
        language="en",
        question_id="q",
        tags=[],
    )


def test_masked_soft_targets_have_finite_loss_and_gradients():
    logits = torch.tensor([[0.0, 1.0, -torch.inf], [1.0, 1.0, 1.0]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    targets = torch.tensor([[0.0, 1.0, 0.0], [0.2, 0.3, 0.5]])
    loss = distribution_loss(logits, targets, valid, torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_metrics_handle_soft_labels_and_expected_scores():
    samples = [sample([1.0, 0.0]), sample([0.0, 1.0]), sample([0.5, 0.5])]
    result = summarize_predictions(samples, [torch.zeros(2)] * 3)
    assert result["nll"] == pytest.approx(math.log(2))
    assert result["expected_accuracy"] == 0.5
    assert result["hard_label_accuracy"] == 0.5
    assert result["ece_10_bins"] == 0
    ordinal = summarize_predictions([sample([0.0, 0.0, 1.0], 1)], [torch.zeros(3)])
    assert ordinal["score_mae"] == 1


def test_collate_variable_candidates():
    batch = collate([sample([0.0, 1.0]), sample([1.0, 0.0, 0.0, 0.0])])
    assert batch["features"].shape == (2, 4, 12)
    assert batch["valid"].sum().item() == 6
    assert batch["targets"].sum().item() == 2
