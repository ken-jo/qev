import torch
from torch import nn

from veyra.option_model import LoRALinear, OptionConfig, OptionModel, option_prompt
from veyra.policy_data import typed_decision
from veyra.schema import DecisionRequest


def test_lora_starts_as_identity_then_merges_exactly():
    torch.manual_seed(9)
    base = nn.Linear(8, 6, bias=False)
    layer = LoRALinear(base, rank=2, alpha=4)
    inputs = torch.randn(4, 8)
    assert torch.equal(layer(inputs), base(inputs))
    layer(inputs).sum().backward()
    assert layer.lora_b.grad.abs().sum() > 0
    assert base.weight.grad is None
    with torch.no_grad():
        layer.lora_b.add_(torch.randn_like(layer.lora_b) * 0.1)
    expected = layer(inputs)
    assert torch.allclose(layer.merge()(inputs), expected, atol=1e-6)


def test_precise_merge_accumulates_before_bfloat16_rounding():
    base = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    layer = LoRALinear(base, rank=2, alpha=4)
    with torch.no_grad():
        layer.lora_b.copy_(torch.randn_like(layer.lora_b) * 0.03)
        expected = (base.weight.float() + layer.lora_b @ layer.lora_a * layer.scale).bfloat16()
        merged = layer.merge("float32")
    assert torch.equal(merged.weight, expected)


def test_option_prompt_ignores_choice_ids_and_order():
    request = DecisionRequest.model_validate(
        {
            "state": {"text": "Observation: 3 kg."},
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Choose the desk.",
                    "criteria": {"x": "regular", "y": "freight"},
                }
            },
        }
    )
    first, positions = option_prompt(request, request.questions["q"])
    changed = request.model_dump()
    changed["questions"]["q"]["criteria"] = {"other_y": "freight", "other_x": "regular"}
    second = DecisionRequest.model_validate(changed)
    text, new_positions = option_prompt(second, second.questions["q"])
    assert first == text
    assert positions == [1, 0]
    assert new_positions == [0, 1]
    assert OptionConfig().encoding == "option-v1"
    rotated, rotated_positions = option_prompt(request, request.questions["q"], rotation=1)
    assert "A. regular" in rotated
    assert "B. freight" in rotated
    assert rotated_positions == [0, 1]


def test_counterfactual_targets_follow_meanings_for_all_types():
    import random

    for kind in ("choice", "score", "noul"):
        for answer in ("desk a", "desk b"):
            question, targets = typed_decision(
                random.Random(1), "Apply the policy.", ["desk a", "desk b"], answer, kind, "desk a"
            )
            assert sum(targets.values()) == 1
            if kind == "noul":
                assert targets["true"] == float(answer == "desk a")
            elif kind == "score":
                assert targets[str(["desk a", "desk b"].index(answer))] == 1
            else:
                assert question["criteria"][max(targets, key=targets.get)] == answer


def test_option_checkpoint_roundtrip_and_integrity(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import pytest

    from veyra.backbone import QwenEncoder
    from veyra.model import VeyraModel

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
    model = OptionModel(fake_load(OptionConfig()))
    model.save(tmp_path, {"completed": True, "optimizer_steps": 3})
    restored = VeyraModel.load(tmp_path, device="cpu")
    assert isinstance(restored, OptionModel)
    assert restored.trained
    assert restored.decision_views == 1
    assert torch.equal(model.readout.weight, restored.readout.weight)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["backbone"]["revision"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        OptionModel.load(tmp_path, device="cpu")


def test_view_consensus_maps_each_internal_order_back_to_candidate_meanings():
    import threading

    # Exercise the readout aggregation independently of a costly backbone load.
    model = OptionModel.__new__(OptionModel)
    nn.Module.__init__(model)
    model._request_lock = threading.RLock()
    model.readout = nn.Identity()
    model.encode = lambda *_args: (
        torch.tensor([[3.0, 1.0], [2.0, 4.0]]),
        {("q", 0): [0, 1], ("q", 1): [1, 0]},
        200,
    )
    logits, tokens = model(None)
    assert torch.equal(logits["q"], torch.tensor([3.5, 1.5]))
    assert tokens == 200
