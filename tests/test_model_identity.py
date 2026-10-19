import pytest
import torch
from torch import nn

from veyra.constants import MODEL_ID, MODEL_REVISION
from veyra.model import VeyraModel
from veyra.option_model import OptionModel
from veyra.probability import Calibration
from veyra.schema import DecisionRequest


@pytest.mark.parametrize("model_class", [VeyraModel, OptionModel])
def test_predict_reports_qev_and_separate_backbone_provenance(model_class, monkeypatch):
    # Exercise the real prediction serializers with fixed CPU logits, without loading weights.
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.trained = True
    model.calibration = Calibration()
    model.decision_views = 1
    model.backbone = {"model_id": "ken-jo/QEV-0.8B", "revision": "a" * 40}
    monkeypatch.setattr(
        model_class, "forward", lambda *_args, **_kwargs: ({"q": torch.tensor([0.0, 1.0])}, 12)
    )
    request = DecisionRequest.model_validate(
        {
            "state": {"text": "Evidence."},
            "questions": {"q": {"type": "noul", "instructions": "True?"}},
        }
    )
    response = model.predict(request)
    assert response["model"] == "qev"
    assert response["backbone"] == (
        model.backbone
        if model_class is OptionModel
        else {"model_id": MODEL_ID, "revision": MODEL_REVISION}
    )
