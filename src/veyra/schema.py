"""Public typed request contract; candidate identities never define the head width."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from veyra.constants import MODEL_ID

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^\S(?:.*\S)?$")
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ImageInput(StrictModel):
    path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]


class State(StrictModel):
    text: Annotated[str, StringConstraints(max_length=100_000)] = ""
    images: list[ImageInput] = Field(default_factory=list, max_length=1)

    @model_validator(mode="after")
    def has_content(self) -> State:
        if not self.text.strip() and not self.images:
            raise ValueError("state must contain nonempty text or an image")
        return self


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Text
    criteria: dict[Identifier, Text] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def distinct_meanings(self) -> ChoiceQuestion:
        if len(set(self.criteria.values())) != len(self.criteria):
            raise ValueError("choice descriptions must be distinct")
        return self


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Text
    criteria: list[Text] = Field(min_length=2, max_length=16)


class NoulCriteria(StrictModel):
    true: Text = "The proposition is true."
    false: Text = "The proposition is false."


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Text
    criteria: NoulCriteria = Field(default_factory=NoulCriteria)


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class DecisionRequest(StrictModel):
    model: Literal[MODEL_ID] = MODEL_ID
    state: State
    questions: dict[Identifier, Question] = Field(min_length=1, max_length=4)

    @classmethod
    def from_json(cls, text: str) -> DecisionRequest:
        """Reject duplicate keys before a JSON parser silently discards an alternative."""

        def unique_object(pairs: list[tuple[str, object]]) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        return cls.model_validate(json.loads(text, object_pairs_hook=unique_object))
