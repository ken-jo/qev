import torch
from test_training import sample

from veyra.calibrate import fit_abstention, fit_temperature
from veyra.probability import Calibration, typed_answer
from veyra.schema import NoulQuestion


def test_overconfident_errors_increase_temperature():
    samples = [sample([1.0, 0.0]), sample([0.0, 1.0])] * 12
    logits = [torch.tensor([6.0, 0.0])] * 24
    assert fit_temperature(samples, logits) > 1


def test_abstention_rejects_everything_when_no_policy_meets_error():
    samples = [sample([0.0, 1.0])] * 24
    result = fit_abstention(samples, [torch.tensor([100.0, 0.0])] * 24, 1.0)
    assert result["always_abstain"] is True
    answer = typed_answer(
        NoulQuestion(type="noul", instructions="True?"),
        torch.tensor([0.0, 100.0]),
        Calibration(always_abstain_types=("noul",)),
    )
    assert answer["confidence"] == 1.0
    assert answer["abstained"] is True


def test_policy_accepts_correct_high_confidence_group():
    samples = [sample([1.0, 0.0])] * 24 + [sample([0.0, 1.0])] * 24
    logits = [torch.tensor([4.0, 0.0])] * 24 + [torch.tensor([0.1, 0.0])] * 24
    result = fit_abstention(samples, logits, 1.0)
    assert result["accepted"] == 24
    assert result["observed_error"] == 0
    assert result["threshold"] > 0.9
