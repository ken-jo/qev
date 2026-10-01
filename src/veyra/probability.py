"""Typed serialization and explicit calibration/abstention state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import torch
from torch import Tensor

from veyra.candidates import candidates_for
from veyra.constants import QUESTION_TYPES
from veyra.schema import ChoiceQuestion, NoulQuestion, Question


@dataclass(frozen=True)
class Calibration:
    temperatures: tuple[float, float, float] = (1.0, 1.0, 1.0)
    abstain_thresholds: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fitted_types: tuple[str, ...] = ()
    abstention_fitted_types: tuple[str, ...] = ()
    always_abstain_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.temperatures) != 3 or len(self.abstain_thresholds) != 3:
            raise ValueError("calibration requires one value per question type")
        if any(not isfinite(x) or x <= 0 for x in self.temperatures):
            raise ValueError("temperatures must be finite and positive")
        if any(not isfinite(x) or not 0 <= x <= 1 for x in self.abstain_thresholds):
            raise ValueError("abstention thresholds must be in [0, 1]")
        if set(self.fitted_types + self.abstention_fitted_types + self.always_abstain_types) - set(
            QUESTION_TYPES
        ):
            raise ValueError("unknown calibrated type")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Calibration:
        return cls(**{key: tuple(value) for key, value in data.items()})


def typed_answer(question: Question, logits: Tensor, calibration: Calibration) -> dict:
    candidates = candidates_for(question)
    if logits.ndim != 1 or logits.numel() != len(candidates) or not torch.isfinite(logits).all():
        raise ValueError("expected one finite logit per supplied candidate")
    type_id = QUESTION_TYPES.index(question.type)
    probs = torch.softmax(logits.detach().float() / calibration.temperatures[type_id], dim=-1)
    values = probs.cpu().tolist()
    winner = int(probs.argmax())
    confidence = values[winner]
    answer = {
        "type": question.type,
        "probabilities": dict(zip((c.key for c in candidates), values, strict=True)),
        "confidence": confidence,
        "confidence_definition": "maximum_candidate_probability",
        "calibrated": question.type in calibration.fitted_types,
        "abstained": (
            question.type in calibration.always_abstain_types
            or confidence < calibration.abstain_thresholds[type_id]
        ),
        "abstention_policy_fitted": question.type in calibration.abstention_fitted_types,
    }
    if isinstance(question, ChoiceQuestion):
        answer["choice"] = candidates[winner].key
    elif isinstance(question, NoulQuestion):
        answer["noul"] = values[1]
    else:
        answer["score"] = sum(i * p for i, p in enumerate(values))
        answer["legend"] = {c.key: c.description for c in candidates}
    return answer
