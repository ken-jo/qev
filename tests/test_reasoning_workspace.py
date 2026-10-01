import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from veyra.backbone import QwenEncoder
from veyra.option_model import OptionConfig, OptionModel
from veyra.reasoning_workspace import condition_positions, workspace_suffix


def test_internal_marker_uses_appended_occurrence_and_respects_padding():
    ids = torch.tensor([[9, 7, 8, 4, 7, 8, 0], [0, 0, 9, 4, 7, 8, 3]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1]])
    assert condition_positions(ids, mask, [7, 8]).tolist() == [3, 3]
    with pytest.raises(ValueError):
        condition_positions(ids, mask, [88, 89])


@pytest.mark.parametrize("slots", [0, -1, 33, True])
def test_workspace_size_is_bounded(slots):
    with pytest.raises(ValueError):
        workspace_suffix(slots)


def test_condition_and_final_readouts_share_one_encoding_call():
    model = OptionModel.__new__(OptionModel)
    nn.Module.__init__(model)
    model._request_lock = threading.RLock()
    model.readout = nn.Identity()
    model.condition_readout = nn.Identity()
    calls = []

    def encode(*args, **kwargs):
        calls.append(kwargs)
        return torch.tensor([[1.0, 2.0]]), {"q": [1, 0]}, 50, torch.tensor([[3.0, 1.0, 0.0]])

    model.encode = encode
    answers, tokens, conditions = model.forward_with_condition(None)
    assert calls == [{"capture_condition": True}]
    assert tokens == 50
    assert answers["q"].tolist() == [2.0, 1.0]
    assert conditions["q"].tolist() == [3.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "binding_rank,binding_mode", [(0, "condition"), (4, "condition"), (4, "uniform")]
)
def test_workspace_checkpoint_roundtrip(tmp_path, monkeypatch, binding_rank, binding_mode):
    def fake_load(config, *_args, **_kwargs):
        base = nn.Module()
        base.language_model = nn.Module()
        base.language_model.layers = nn.ModuleList([nn.Linear(1, 1)])
        base.language_model.embed_tokens = nn.Embedding(16, 2048)
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda value, **kwargs: [ord(value) - 65])
        )
        return QwenEncoder(base, processor, config)

    monkeypatch.setattr(QwenEncoder, "load", fake_load)
    model = OptionModel(
        fake_load(OptionConfig()),
        reasoning_slots=4,
        binding_rank=binding_rank,
        binding_mode=binding_mode,
    )
    if binding_rank:
        with torch.no_grad():
            model.binding_head.experts.weight.normal_()
    model.save(tmp_path, {"completed": True, "optimizer_steps": 3})
    restored = OptionModel.load(tmp_path, device="cpu")
    assert restored.reasoning_slots == 4
    assert restored.binding_rank == binding_rank
    assert restored.binding_mode == binding_mode
    assert torch.equal(model.condition_readout.weight, restored.condition_readout.weight)
    if binding_rank:
        for name, value in model.binding_head.state_dict().items():
            assert torch.equal(value, restored.binding_head.state_dict()[name])


def test_binding_inference_uses_the_same_single_forward_as_diagnostics():
    from veyra.binding_head import ConditionedReadout

    model = OptionModel.__new__(OptionModel)
    nn.Module.__init__(model)
    model._request_lock = threading.RLock()
    model.binding_rank = 4
    model.readout = nn.Linear(2, 16, bias=False)
    model.condition_readout = nn.Identity()
    model.binding_head = ConditionedReadout(rank=4, input_size=2).eval()
    with torch.no_grad():
        model.binding_head.experts.weight.normal_()
    calls = []

    def encode(*_args, **kwargs):
        calls.append(kwargs)
        return torch.tensor([[1.0, 2.0]]), {"q": [1, 0]}, 50, torch.tensor([[3.0, 1.0, 0.0]])

    model.encode = encode
    normal, normal_tokens = model(None)
    assert calls == [{"capture_condition": True}]
    calls.clear()
    diagnostic, diagnostic_tokens, conditions = model.forward_with_condition(None)
    assert calls == [{"capture_condition": True}]
    assert normal_tokens == diagnostic_tokens == 50
    assert torch.equal(normal["q"], diagnostic["q"])
    assert conditions["q"].shape == (3,)
