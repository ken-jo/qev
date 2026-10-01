import json

import pytest
from pydantic import ValidationError

from veyra.candidates import candidates_for, render_candidate
from veyra.schema import DecisionRequest, State


def request(criteria=None, **question):
    return {
        "state": {"text": "A blue parcel is visible."},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which description matches the image?",
                "criteria": criteria or {"a": "blue parcel", "b": "red parcel"},
                **question,
            }
        },
    }


def test_choice_ids_do_not_enter_model_input():
    a = DecisionRequest.model_validate(request())
    b = DecisionRequest.model_validate(
        request({"unseen_1": "blue parcel", "unseen_2": "red parcel"})
    )
    for ca, cb in zip(candidates_for(a.questions["route"]), candidates_for(b.questions["route"])):
        assert render_candidate(a.state, a.questions["route"], ca) == render_candidate(
            b.state, b.questions["route"], cb
        )
        assert ca.key != cb.key


@pytest.mark.parametrize("count", [2, 3, 8, 16])
def test_dynamic_cardinality(count):
    parsed = DecisionRequest.model_validate(request({str(i): f"meaning {i}" for i in range(count)}))
    assert len(candidates_for(parsed.questions["route"])) == count


@pytest.mark.parametrize("count", [1, 17])
def test_invalid_cardinality_rejected(count):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(request({str(i): f"meaning {i}" for i in range(count)}))


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        DecisionRequest.from_json('{"state": {}, "state": {"text": "duplicate"}}')


def test_empty_state_rejected():
    with pytest.raises(ValidationError):
        State(text="  ")


def test_image_only_state_supported():
    assert State(images=[{"path": "image.png"}]).text == ""


def test_score_preserves_ordinal_meaning():
    parsed = DecisionRequest.model_validate(
        request(type="score", criteria=["none", "some", "many"])
    )
    options = candidates_for(parsed.questions["route"])
    assert [c.key for c in options] == ["0", "1", "2"]
    assert [c.level for c in options] == [0, 0.5, 1]
    prompt = json.loads(render_candidate(parsed.state, parsed.questions["route"], options[2]))
    assert prompt["candidate"]["ordinal_level"] == 2


def test_noul_defaults_and_custom_criteria():
    parsed = DecisionRequest.model_validate(
        {"state": {"text": "blue"}, "questions": {"q": {"type": "noul", "instructions": "Blue?"}}}
    )
    assert [c.key for c in candidates_for(parsed.questions["q"])] == ["false", "true"]


def test_wrong_backbone_and_extra_fields_rejected():
    for changed in [{"model": "another-model"}, {"unrecognized": True}]:
        with pytest.raises(ValidationError):
            DecisionRequest.model_validate({**request(), **changed})


def test_questions_change_rendered_input():
    a = DecisionRequest.model_validate(request())
    b = DecisionRequest.model_validate(request(instructions="Which parcel should be inspected?"))
    ca = candidates_for(a.questions["route"])[0]
    assert render_candidate(a.state, a.questions["route"], ca) != render_candidate(
        b.state, b.questions["route"], ca
    )
