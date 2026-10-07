import json

import pytest
from fastapi.testclient import TestClient

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
