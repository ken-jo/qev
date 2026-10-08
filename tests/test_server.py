import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from veyra.constants import MODEL_ID
from veyra.server import create_app


class StubTrainedModel:
    trained = True

    def predict(self, request, image_root):
        return {"answers": {name: {"type": q.type} for name, q in request.questions.items()}}


def payload():
    return {
        "state": {"text": "hello"},
        "questions": {"named": {"type": "noul", "instructions": "Is this a greeting?"}},
    }


def test_api_contract_and_duplicate_keys(tmp_path):
    client = TestClient(create_app(StubTrainedModel(), tmp_path))
    assert client.get("/health").json()["trained"] is True
    assert client.get("/v1/schema").status_code == 200
    assert client.post("/v1/systemone", json=payload()).json()["answers"]["named"]["type"] == "noul"
    response = client.post("/v1/systemone", content='{"state":{},"state":{}}')
    assert response.status_code == 422
    assert "duplicate JSON key" in response.text


def test_api_image_root_and_request_size(tmp_path):
    client = TestClient(create_app(StubTrainedModel(), tmp_path))
    request = payload()
    request["state"]["images"] = [{"path": "../outside.png"}]
    assert client.post("/v1/systemone", json=request).status_code == 422
    assert client.post("/v1/systemone", content=b"x" * (1024 * 1024 + 1)).status_code == 413
    assert client.post("/v1/systemone", json={"unknown": "field"}).status_code == 422


def test_untrained_server_rejected(tmp_path):
    model = StubTrainedModel()
    model.trained = False
    with pytest.raises(ValueError, match="trained"):
        create_app(model, tmp_path)


def test_image_decode_error_is_a_client_error(tmp_path):
    from PIL import UnidentifiedImageError

    class InvalidImageModel(StubTrainedModel):
        def predict(self, request, image_root):
            raise UnidentifiedImageError("invalid image")

    client = TestClient(create_app(InvalidImageModel(), tmp_path))
    assert client.post("/v1/systemone", json=payload()).status_code == 422


def test_systemone_compatible_bodies_are_normalized(tmp_path):
    seen = []

    class Recording(StubTrainedModel):
        def predict(self, request, image_root):
            seen.append(request)
            return super().predict(request, image_root)

    client = TestClient(create_app(Recording(), tmp_path))
    body = {
        "model": "qev",
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0}},
        "questions": {
            "large": {"type": "noul"},
            "urgency": {"type": "score", "criteria": ["Can wait", "Today"]},
        },
    }
    assert client.post("/v1/systemone", json=body).status_code == 200
    request = seen[-1]
    assert json.loads(request.state.text) == {"invoice": {"vendor": "Acme", "total": 1250.0}}
    assert request.questions["large"].instructions == "large"
    body["state"] = "Checkout is returning errors."
    assert client.post("/v1/systemone", json=body).status_code == 200
    assert seen[-1].state.text == "Checkout is returning errors."


def test_systemone_normalization_keeps_native_requests_and_rejections(tmp_path):
    client = TestClient(create_app(StubTrainedModel(), tmp_path))
    native = payload()
    assert client.post("/v1/systemone", json=native).status_code == 200
    assert client.post("/v1/systemone", json={**native, "model": "other"}).status_code == 422
    assert client.post("/v1/systemone", json={**native, "state": ""}).status_code == 422
    assert client.post("/v1/systemone", json={**native, "state": {}}).status_code == 422
    assert client.post("/v1/systemone", json={**native, "extra": 1}).status_code == 422


@pytest.fixture
def recording_model():
    model = Mock(spec=StubTrainedModel, trained=True)
    model.predict.return_value = {"answers": {}}
    return model


@pytest.mark.parametrize(
    "state",
    [
        {"images": [{"path": "item.jpg"}], "context": "Inspect damage"},
        {"text": "Inspect damage", "context": "A photograph is attached"},
        {
            "text": "Inspect damage",
            "images": [{"path": "item.jpg"}],
            "context": "A photograph is attached",
        },
        {"text": 125, "invoice": {"total": 1250}},
        {"images": "item.jpg", "invoice": {"total": 1250}},
    ],
)
def test_systemone_native_state_fields_remain_strict(tmp_path, recording_model, state):
    client = TestClient(create_app(recording_model, tmp_path))
    response = client.post(
        "/v1/systemone",
        json={
            "model": "qev",
            "state": state,
            "questions": {"Is the item damaged?": {"type": "noul"}},
        },
    )
    assert response.status_code == 422
    assert any(error["type"] == "extra_forbidden" for error in response.json()["detail"])
    recording_model.predict.assert_not_called()


@pytest.mark.parametrize(
    ("state", "instructions"),
    [
        ({"images": [{"path": "item.jpg"}]}, None),
        (
            {"text": "Inspect damage", "images": [{"path": "item.jpg"}]},
            "Read the photograph for visible damage.",
        ),
    ],
)
def test_systemone_native_images_are_preserved(tmp_path, recording_model, state, instructions):
    client = TestClient(create_app(recording_model, tmp_path))
    question = {"type": "noul"}
    if instructions is not None:
        question["instructions"] = instructions
    body = {"model": "qev", "state": state, "questions": {"Is the item damaged?": question}}
    assert client.post("/v1/systemone", json=body).status_code == 200
    recording_model.predict.assert_called_once()
    request = recording_model.predict.call_args.args[0]
    assert request.state.text == state.get("text", "")
    assert [image.path for image in request.state.images] == ["item.jpg"]
    assert request.model == MODEL_ID
    assert request.questions["Is the item damaged?"].instructions == (
        instructions or "Is the item damaged?"
    )
    assert recording_model.predict.call_args.kwargs["image_root"] == tmp_path.resolve()


@pytest.mark.parametrize(
    "state",
    [
        {"invoice": {"customer": "고객", "paid": True, "total": 1250.0}},
        [{"paid": True}, 7],
        False,
        0,
        1.25,
    ],
)
def test_systemone_business_json_states_are_serialized(tmp_path, recording_model, state):
    client = TestClient(create_app(recording_model, tmp_path))
    body = {"model": "qev", "state": state, "questions": {"Paid?": {"type": "noul"}}}
    assert client.post("/v1/systemone", json=body).status_code == 200
    recording_model.predict.assert_called_once()
    request = recording_model.predict.call_args.args[0]
    assert request.state.text == json.dumps(state, ensure_ascii=False)
    assert request.state.images == []
    assert request.questions["Paid?"].instructions == "Paid?"


@pytest.mark.parametrize("state", [None, "", " \t\n", {}, {"text": ""}, {"images": []}])
def test_systemone_null_and_empty_states_rejected(tmp_path, recording_model, state):
    client = TestClient(create_app(recording_model, tmp_path))
    body = {**payload(), "model": "qev", "state": state}
    assert client.post("/v1/systemone", json=body).status_code == 422
    recording_model.predict.assert_not_called()


def test_systemone_missing_state_rejected(tmp_path, recording_model):
    client = TestClient(create_app(recording_model, tmp_path))
    body = {"model": "qev", "questions": {"Paid?": {"type": "noul"}}}
    assert client.post("/v1/systemone", json=body).status_code == 422
    recording_model.predict.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        '{"state":{"invoice":{"vendor":"Acme","vendor":"Other"}},'
        '"questions":{"Paid?":{"type":"noul"}}}',
        '{"state":{"images":[{"path":"item.jpg","path":"other.jpg"}]},'
        '"questions":{"Damaged?":{"type":"noul"}}}',
    ],
)
def test_systemone_nested_duplicate_keys_rejected(tmp_path, recording_model, body):
    client = TestClient(create_app(recording_model, tmp_path))
    response = client.post("/v1/systemone", content=body)
    assert response.status_code == 422
    assert "duplicate JSON key" in response.text
    recording_model.predict.assert_not_called()
