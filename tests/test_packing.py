import pytest

from veyra.packing import marker_positions, pack_request
from veyra.schema import DecisionRequest


def make_request(criteria):
    return DecisionRequest.model_validate(
        {
            "state": {"text": "Evidence."},
            "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": criteria}},
        }
    )


def test_choice_order_and_ids_do_not_change_backbone_prompt():
    a = pack_request(make_request({"x": "red", "y": "blue", "z": "green"}))
    b = pack_request(make_request({"1": "green", "2": "red", "3": "blue"}))
    assert a.text == b.text
    assert a.layouts["q"].candidate_indices == (2, 0, 1)
    assert b.layouts["q"].candidate_indices == (1, 2, 0)


def test_same_semantic_question_encodes_once():
    request = make_request({"x": "red", "y": "blue"})
    request.questions["alias"] = request.questions["q"].model_copy(deep=True)
    packed = pack_request(request)
    assert packed.question_count == 1
    assert packed.candidate_count == 2
    assert set(packed.layouts) == {"q", "alias"}


def test_marker_alignment_and_collision():
    assert marker_positions([1, 2, 3, 7, 2, 3], [2, 3], 2) == [2, 5]
    with pytest.raises(ValueError, match="alignment mismatch"):
        marker_positions([1, 2], [5], 1)
    request = make_request({"x": "red", "y": "blue"})
    request.state.text = "[VEYRA_CANDIDATE]"
    with pytest.raises(ValueError, match="reserved"):
        pack_request(request)


def test_duplicate_choice_descriptions_rejected():
    with pytest.raises(ValueError, match="distinct"):
        make_request({"x": "same", "y": "same"})
