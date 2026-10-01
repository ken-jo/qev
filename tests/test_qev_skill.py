import copy
import importlib.util
import json
from pathlib import Path

import pytest

from qev import DecisionRequest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qev_skill_decide", ROOT / "skills/qev/scripts/decide.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def fixture():
    request = DecisionRequest.from_json(
        (ROOT / "skills/qev/assets/photograph/request.json").read_text("utf-8")
    )
    response = json.loads((ROOT / "examples/photograph/response.json").read_text("utf-8"))
    return request, response


def test_skill_preserves_three_typed_probabilities():
    request, response = fixture()
    result = helper.summarize(request, response)
    assert result["response"] is response
    assert result["model_called"] is True
    assert set(item["type"] for item in result["decisions"].values()) == {"choice", "score", "noul"}


def test_skill_cannot_relax_model_abstention():
    request, response = fixture()
    response["answers"]["choice"]["abstained"] = True
    result = helper.summarize(request, response, min_confidence=0)
    assert result["status"] == "review"
    assert result["decisions"]["choice"]["reasons"] == ["model_abstained"]


def test_skill_can_add_explicit_review_policy():
    request, response = fixture()
    label = response["answers"]["choice"]["choice"]
    result = helper.summarize(request, response, review_labels=[label])
    assert result["status"] == "review"
    assert "explicit_review_candidate" in result["decisions"]["choice"]["reasons"]
    result = helper.summarize(request, response, min_confidence=1)
    assert result["status"] == "review"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(model="other-model"),
        lambda r: r.update(backbone=[]),
        lambda r: r["answers"].pop("choice"),
        lambda r: r["answers"]["choice"].update(choice=[]),
        lambda r: r["answers"]["choice"].update(abstained="false"),
        lambda r: r["answers"]["choice"].update(confidence=float("nan")),
        lambda r: r["answers"]["score"].update(score=100),
        lambda r: r["answers"]["noul"].update(noul=2),
        lambda r: r["answers"]["choice"]["probabilities"].update(unknown=0.1),
    ],
)
def test_skill_rejects_invalid_server_response(mutation):
    request, response = fixture()
    response = copy.deepcopy(response)
    mutation(response)
    with pytest.raises(ValueError):
        helper.summarize(request, response)


def test_skill_dry_run_never_calls_model(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        raise AssertionError("Dry run must not send a model request")

    monkeypatch.setattr(helper, "http_predict", unexpected)
    assert (
        helper.main(["--request", str(ROOT / "skills/qev/assets/text-request.json"), "--dry-run"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["model_called"] is False
    assert "response" not in result


def test_skill_transport_failure_has_no_prediction(monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise OSError("test server unavailable")

    monkeypatch.setattr(helper, "http_predict", unavailable)
    assert helper.main(["--request", str(ROOT / "skills/qev/assets/text-request.json")]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "unavailable" in output.err
