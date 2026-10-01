import pytest
import torch
from test_training import sample

from veyra.training import summarize_predictions


def test_uniform_probability_metrics_and_soft_labels():
    item = sample([0.5, 0.5])
    metrics = summarize_predictions([item], [torch.zeros(2)])
    assert metrics["nll"] == pytest.approx(0.69314718056)
    assert metrics["brier"] == 0
    assert metrics["expected_accuracy"] == 0.5
    assert metrics["hard_label_accuracy"] is None
    assert metrics["ece_10_bins"] == 0


def test_temperature_preserves_argmax_but_changes_probability_quality():
    item = sample([0.0, 1.0])
    first = summarize_predictions([item], [torch.tensor([0.0, 2.0])])
    second = summarize_predictions([item], [torch.tensor([0.0, 2.0])], (2, 2, 2))
    assert first["hard_label_accuracy"] == second["hard_label_accuracy"] == 1
    assert first["nll"] < second["nll"]
