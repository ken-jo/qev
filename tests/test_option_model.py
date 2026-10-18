import torch
from torch import nn

from veyra.constants import MODEL_ID, MODEL_REVISION
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
        base.language_model.config = SimpleNamespace(hidden_size=2048)
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


def fake_exit_backbone(config, layers=4, hidden=2048):
    from types import SimpleNamespace

    from veyra.backbone import QwenEncoder

    base = nn.Module()
    base.language_model = nn.Module()
    base.language_model.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(layers)])
    base.language_model.norm = nn.LayerNorm(hidden)
    base.language_model.embed_tokens = nn.Embedding(16, hidden)
    base.language_model.config = SimpleNamespace(
        hidden_size=hidden,
        num_hidden_layers=layers,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda value, **kwargs: [ord(value) - 65])
    )
    return QwenEncoder(base, processor, config)


def test_exit_head_width_follows_the_backbone_hidden_size():
    from veyra.exit_head import ExitHead

    for hidden in (1024, 2048):
        head = ExitHead(hidden)
        assert head.n.normalized_shape == (2 * hidden,)
        logits = head(torch.randn(3, hidden), torch.randn(3, hidden))
        assert logits.shape == (3, 16)


def test_early_exit_manifest_block_is_validated():
    import pytest

    from veyra.exit_head import parse_early_exit

    assert parse_early_exit({"layer": 16, "head": "exit16.safetensors"}, 24) == (
        16,
        "exit16.safetensors",
        None,
    )
    for block in (
        {"layer": 0, "head": "exit.safetensors"},
        {"layer": 25, "head": "exit.safetensors"},
        {"layer": True, "head": "exit.safetensors"},
        {"layer": 16, "head": "../exit.safetensors"},
        {"layer": 16, "head": "exit16.pt"},
        {"layer": 16},
        {"layer": 16, "head": "exit.safetensors", "extra": 1},
    ):
        with pytest.raises(ValueError, match="early_exit"):
            parse_early_exit(block, 24)


def test_early_exit_is_opt_in_and_truncates_the_text_stack(tmp_path, monkeypatch):
    import hashlib
    import json

    import pytest
    from safetensors.torch import save_file

    from veyra.backbone import QwenEncoder
    from veyra.exit_head import ExitHead

    monkeypatch.setattr(QwenEncoder, "load", lambda config, *_a, **_k: fake_exit_backbone(config))
    model = OptionModel(fake_exit_backbone(OptionConfig()), reasoning_slots=4)
    model.save(tmp_path, {"completed": True, "optimizer_steps": 3})
    with pytest.raises(ValueError, match="no early_exit"):
        OptionModel.load(tmp_path, device="cpu", early_exit=True)

    torch.manual_seed(3)
    head = ExitHead(2048)
    save_file(head.state_dict(), tmp_path / "exit2.safetensors")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["early_exit"] = {"layer": 2, "head": "exit2.safetensors"}
    manifest_path.write_text(json.dumps(manifest))

    default = OptionModel.load(tmp_path, device="cpu")
    assert default.exit_head is None
    assert default.encoder.config.fast_jpeg_decode is False
    assert "fast_jpeg_decode" not in manifest["encoder"]
    assert len(default.encoder.model.language_model.layers) == 4
    assert isinstance(default.encoder.model.language_model.norm, nn.LayerNorm)

    exiting = OptionModel.load(tmp_path, device="cpu", early_exit=True)
    language_model = exiting.encoder.model.language_model
    assert exiting.early_exit_layer == 2
    assert len(language_model.layers) == 2
    assert language_model.config.num_hidden_layers == 2
    assert language_model.config.layer_types == ["linear_attention", "linear_attention"]
    assert isinstance(language_model.norm, nn.Identity)
    for name, value in head.state_dict().items():
        assert torch.equal(exiting.exit_head.state_dict()[name], value)
    with pytest.raises(ValueError, match="inference only"):
        exiting.save(tmp_path / "copy", {"completed": True, "optimizer_steps": 1})

    manifest["early_exit"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        OptionModel.load(tmp_path, device="cpu", early_exit=True)
    digest = hashlib.sha256((tmp_path / "exit2.safetensors").read_bytes()).hexdigest()
    manifest["early_exit"]["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))
    assert OptionModel.load(tmp_path, device="cpu", early_exit=True).early_exit_layer == 2

    save_file(ExitHead(1024).state_dict(), tmp_path / "exit2.safetensors")
    del manifest["early_exit"]["sha256"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hidden size"):
        OptionModel.load(tmp_path, device="cpu", early_exit=True)


def test_early_exit_requires_the_condition_position():
    import pytest

    from veyra.exit_head import ExitHead

    model = OptionModel(fake_exit_backbone(OptionConfig()))
    with pytest.raises(ValueError, match="condition position"):
        model.enable_early_exit(2, ExitHead(2048).state_dict())


def test_early_exit_forward_reads_answer_and_condition_states():
    import threading

    from veyra.exit_head import ExitHead

    torch.manual_seed(5)
    model = OptionModel.__new__(OptionModel)
    nn.Module.__init__(model)
    model._request_lock = threading.RLock()
    model.exit_head = ExitHead(8)
    model.readout = None  # the full-depth readout must not be used
    states, conditions = torch.randn(2, 8), torch.randn(2, 8)
    model.encode = lambda *_args, capture_condition=False: (
        (
            states,
            {"a": [2, 0], "b": [1, 3]},
            90,
        )
        + ((conditions,) if capture_condition else ())
    )
    logits, tokens = model(None)
    expected = model.exit_head(states, conditions)
    assert torch.equal(logits["a"], expected[0, [2, 0]])
    assert torch.equal(logits["b"], expected[1, [1, 3]])
    assert tokens == 90


def test_truncated_qwen_stack_reproduces_the_distilled_layer_output():
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    from veyra.exit_head import truncate_text_stack

    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    torch.manual_seed(0)
    model = Qwen3_5TextModel(config).eval()
    ids = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        full = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        truncate_text_stack(model, 2)
        short = model(input_ids=ids, use_cache=False).last_hidden_state
    # hidden_states[0] is the embedding output, so index L is the raw residual after L layers.
    assert torch.allclose(full[2], short, atol=1e-6)


def head_shapes(hidden):
    """trainable_state shapes of a reasoning-slot, rank-64 binding option head."""
    return {
        "readout.weight": (16, hidden),
        "condition_readout.weight": (3, hidden),
        "binding_head.normalize.weight": (hidden,),
        "binding_head.normalize.bias": (hidden,),
        "binding_head.project.weight": (64, hidden),
        "binding_head.project.bias": (64,),
        "binding_head.experts.weight": (48, 64),
    }


def test_released_2b_head_layout_is_unchanged():
    model = OptionModel(fake_exit_backbone(OptionConfig()), reasoning_slots=4, binding_rank=64)
    state = model.trainable_state()
    assert {name: tuple(value.shape) for name, value in state.items()} == head_shapes(2048)
    assert all(value.dtype == torch.float32 for value in state.values())
    # The readout still starts from the backbone's option-letter embeddings.
    embed = model.encoder.model.language_model.embed_tokens.weight
    assert torch.equal(model.readout.weight, embed[:16].float())
    assert model.backbone == {"model_id": MODEL_ID, "revision": MODEL_REVISION}


def test_head_widths_follow_the_backbone_hidden_size():
    for hidden in (1024, 2048):
        model = OptionModel(
            fake_exit_backbone(OptionConfig(), hidden=hidden), reasoning_slots=4, binding_rank=64
        )
        shapes = {name: tuple(value.shape) for name, value in model.trainable_state().items()}
        assert shapes == head_shapes(hidden)
        states, conditions = torch.randn(2, hidden), torch.randn(2, 3)
        assert model.binding_head(states, conditions).shape == (2, 16)


MERGED = {
    "model_id": "example/qev-small-merged",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "weights": "merged",
}


def test_backbone_manifest_block_is_validated():
    import pytest

    from veyra.option_model import backbone_identity

    default = {"model_id": MODEL_ID, "revision": MODEL_REVISION}
    assert backbone_identity(default) == default
    assert backbone_identity(MERGED) == MERGED
    for block in (
        {"model_id": "example/other", "revision": MERGED["revision"]},
        dict(MERGED, revision="main"),
        dict(MERGED, revision=MERGED["revision"].upper()),
        dict(MERGED, weights="lora"),
        dict(MERGED, model_id=" "),
        dict(MERGED, extra=1),
        {"model_id": MODEL_ID, "revision": "0" * 40},
        None,
    ):
        with pytest.raises(ValueError, match="revision"):
            backbone_identity(block)


def test_merged_backbone_checkpoint_loads_without_lora(tmp_path, monkeypatch):
    import json

    import pytest

    from veyra.backbone import QwenEncoder

    requested = {}

    def fake_load(config, *_args, model_id=MODEL_ID, revision=MODEL_REVISION, **_kwargs):
        requested.update(model_id=model_id, revision=revision)
        return fake_exit_backbone(config, hidden=1024)

    monkeypatch.setattr(QwenEncoder, "load", fake_load)
    model = OptionModel(
        fake_exit_backbone(OptionConfig(), hidden=1024),
        reasoning_slots=4,
        binding_rank=64,
        backbone=MERGED,
    )
    manifest = model.save(tmp_path, {"completed": True, "optimizer_steps": 1})
    assert manifest["backbone"] == MERGED
    manifest_path = tmp_path / "manifest.json"
    for adaptation in (None, {}):
        if adaptation is None:
            manifest.pop("adaptation", None)
        else:
            manifest["adaptation"] = adaptation
        manifest_path.write_text(json.dumps(manifest))
        restored = OptionModel.load(tmp_path, device="cpu")
        assert requested == {"model_id": MERGED["model_id"], "revision": MERGED["revision"]}
        assert restored.adapter_names == [] and restored.adaptation["layers"] == 0
        assert restored.backbone == MERGED
        for name, value in model.trainable_state().items():
            assert torch.equal(restored.trainable_state()[name], value)

    # A 2B-width head cannot be loaded onto a hidden-1024 backbone.
    wide = OptionModel(fake_exit_backbone(OptionConfig()), reasoning_slots=4, binding_rank=64)
    wide.backbone = MERGED
    wide.save(tmp_path / "wide", {"completed": True, "optimizer_steps": 1})
    with pytest.raises(ValueError, match="hidden size"):
        OptionModel.load(tmp_path / "wide", device="cpu")
