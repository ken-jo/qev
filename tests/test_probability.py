import pytest
import torch

from veyra.probability import Calibration, typed_answer
from veyra.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion


def test_choice_ids_and_abstention():
    q = ChoiceQuestion(
        type="choice", instructions="route?", criteria={"novel": "yes", "other": "no"}
    )
    result = typed_answer(q, torch.tensor([0.0, 0.0]), Calibration(abstain_thresholds=(0.7, 0, 0)))
    assert set(result["probabilities"]) == {"novel", "other"}
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["abstained"] is True
    assert result["calibrated"] is False
    assert result["abstention_policy_fitted"] is False


def test_score_expectation_and_noul_direction():
    score = ScoreQuestion(
        type="score", instructions="severity?", criteria=["low", "medium", "high"]
    )
    result = typed_answer(score, torch.log(torch.tensor([0.1, 0.2, 0.7])), Calibration())
    assert result["score"] == pytest.approx(1.6)
    noul = NoulQuestion(type="noul", instructions="true?")
    assert typed_answer(noul, torch.tensor([-2.0, 2.0]), Calibration())["noul"] > 0.98


def test_temperature_and_serialization():
    cfg = Calibration((2, 1, 1), (0.6, 0, 0), ("choice",), ("choice",))
    assert Calibration.from_dict(cfg.to_dict()) == cfg
    q = ChoiceQuestion(type="choice", instructions="route?", criteria={"a": "a", "b": "b"})
    raw = typed_answer(q, torch.tensor([0.0, 4.0]), Calibration())
    calibrated = typed_answer(q, torch.tensor([0.0, 4.0]), cfg)
    assert calibrated["confidence"] < raw["confidence"]
    assert calibrated["calibrated"] is True


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        Calibration((temperature, 1, 1))
