"""Real tiny Qwen3.5 checkpoint loading, integrity failures and legacy regression."""

import hashlib
import json

import pytest
import torch

from veyra.backbone import QwenEncoder
from veyra.bundled_backbone import verify_bundle
from veyra.option_model import OptionConfig, OptionModel
from veyra.schema import DecisionRequest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


@pytest.fixture
def bundled(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3_5Config,
        Qwen3_5Model,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    specials = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": specials})
    tokenizer.pad_token = specials[-1]
    template = (
        "{% for m in messages %}{% for c in m.content %}"
        "{{ c.text | default('') }}{% endfor %}{% endfor %}"
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=4, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=4, merge_size=2),
        tokenizer=tokenizer,
        chat_template=template,
    )
    text = Qwen3_5TextConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"] * 2,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=32,
        patch_size=4,
        num_position_embeddings=16,
    )
    config = Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=tokenizer.convert_tokens_to_ids(specials[0]),
        video_token_id=tokenizer.convert_tokens_to_ids(specials[1]),
        vision_start_token_id=tokenizer.convert_tokens_to_ids(specials[2]),
        vision_end_token_id=tokenizer.convert_tokens_to_ids(specials[3]),
    )
    torch.manual_seed(7)
    backbone = Qwen3_5Model(config).eval()
    root = tmp_path / "backbone"
    backbone.save_pretrained(root)
    processor.save_pretrained(root)
    model = OptionModel(QwenEncoder(backbone, processor, OptionConfig()), reasoning_slots=4)
    manifest = model.save(tmp_path / "head", {"completed": True, "optimizer_steps": 1})
    (tmp_path / "head/head.safetensors").replace(tmp_path / "head.safetensors")
    manifest.update(
        format_version=3,
        adaptation={},
        modes={"accurate": {"image_max_pixels": 262144}},
        default_mode="accurate",
        backbone=dict(
            bundled=True,
            model_id="example/Qwen3.5-tiny",
            revision="a" * 40,
            depth=2,
            vocabulary=len(tokenizer),
            files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
        ),
    )
    write_manifest(tmp_path, manifest)
    return tmp_path, manifest, model


def test_real_bundled_roundtrip_is_offline_and_predicts(bundled, monkeypatch):
    path, manifest, original = bundled
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *_a, **_k: pytest.fail("network")
    )
    restored = OptionModel.load(path, device="cpu", local_files_only=False)
    assert restored.backbone == manifest["backbone"]
    assert restored.serving_mode == "accurate" and not restored.adapter_names
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    request = DecisionRequest.model_validate(
        {
            "state": {"text": "Test evidence"},
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Which?",
                    "criteria": {"a": "alpha", "b": "beta"},
                }
            },
        }
    )
    original.trained = True
    with torch.inference_mode():
        assert restored.predict(request)["answers"] == original.predict(request)["answers"]
    with pytest.raises(ValueError, match="inference only"):
        restored.save(path / "resave", {})


@pytest.mark.parametrize(
    "fault",
    [
        "corrupt",
        "missing",
        "unlisted",
        "head",
        "depth",
        "revision",
        "traversal",
        "adapter",
        "version",
    ],
)
def test_bundled_integrity_fails_before_loading(bundled, monkeypatch, fault):
    path, manifest, _ = bundled
    if fault == "corrupt":
        with (path / "backbone/model.safetensors").open("ab") as f:
            f.write(b"bad")
    elif fault == "missing":
        (path / "backbone/tokenizer.json").unlink()
    elif fault == "unlisted":
        (path / "backbone/extra.json").write_text("{}")
    elif fault == "head":
        (path / "head.safetensors").write_bytes(b"bad")
    elif fault == "depth":
        manifest["backbone"]["depth"] = 3
    elif fault == "revision":
        manifest["backbone"]["revision"] = "main"
    elif fault == "traversal":
        manifest["backbone"]["files"]["../escape"] = "0" * 64
    elif fault == "adapter":
        manifest["adaptation"] = {"layers": 1}
    else:
        manifest["format_version"] = 2
    write_manifest(path, manifest)
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *_a, **_k: pytest.fail("loaded unverified files"),
    )
    with pytest.raises(ValueError):
        OptionModel.load(path, device="cpu")


def test_bundled_fast_is_a_clear_error(bundled):
    path, _, _ = bundled
    with pytest.raises(ValueError, match="unknown serving mode 'fast'.*accurate"):
        OptionModel.load(path, device="cpu", mode="fast")


def test_hub_snapshot_symlink_files_are_verified(bundled, tmp_path):
    path, manifest, _ = bundled
    source = path / "backbone/tokenizer.json"
    blob = tmp_path / "blob"
    source.replace(blob)
    source.symlink_to(blob)
    assert verify_bundle(path, manifest["backbone"]) == path / "backbone"


def test_prefixed_frozen_head_bytes_are_preserved(bundled):
    from safetensors.torch import load_file, save_file

    path, manifest, original = bundled
    head = path / "head.safetensors"
    save_file({"head." + k: v for k, v in load_file(head).items()}, head)
    manifest.update(head_key_prefix="head.", weights_sha256=sha(head))
    write_manifest(path, manifest)
    before = sha(head)
    restored = OptionModel.load(path, device="cpu")
    assert sha(head) == before
    for name, value in original.trainable_state().items():
        torch.testing.assert_close(value, restored.trainable_state()[name], rtol=0, atol=0)


def test_hashed_but_incomplete_weights_are_rejected(bundled):
    from safetensors.torch import load_file, save_file

    path, manifest, _ = bundled
    weights = path / "backbone/model.safetensors"
    state = load_file(weights)
    state.pop(next(iter(state)))
    save_file(state, weights)
    manifest["backbone"]["files"]["model.safetensors"] = sha(weights)
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="weight keys or shapes mismatch"):
        OptionModel.load(path, device="cpu")


@pytest.mark.parametrize("hub", [False, True])
def test_official_engine_accepts_local_or_hub_bundle(bundled, monkeypatch, hub):
    pytest.importorskip("decision_index")  # optional public-board kit
    from qev_decision_index import QEVEngine

    path, _, _ = bundled
    calls = []

    def snapshot(model, revision):
        calls.append((model, revision))
        return str(path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    engine = QEVEngine(
        model="example/compact" if hub else str(path), revision="b" * 40, device="cpu"
    )
    assert engine.m.serving_mode == "accurate"
    assert engine.m.backbone["bundled"] is True
    assert calls == ([("example/compact", "b" * 40)] if hub else [])


@pytest.mark.parametrize("mode", ["accurate", "fast"])
def test_v3_style_manifest_load_path_and_bytes_unchanged(tmp_path, monkeypatch, mode):
    from safetensors.torch import save_file
    from test_option_model import MERGED, MODES, fake_exit_backbone

    from veyra.exit_head import ExitHead

    requested = []

    def legacy_load(config, *args, **kwargs):
        requested.append((config.to_dict(), args, kwargs))
        return fake_exit_backbone(config, hidden=1024)

    monkeypatch.setattr(QwenEncoder, "load", legacy_load)
    model = OptionModel(
        fake_exit_backbone(OptionConfig(), hidden=1024), backbone=MERGED, reasoning_slots=4
    )
    manifest = model.save(tmp_path, {"completed": True, "optimizer_steps": 5419})
    save_file(ExitHead(1024).state_dict(), tmp_path / "exit.safetensors")
    manifest.update(
        modes=MODES,
        default_mode="accurate",
        early_exit=dict(
            layer=2, head="exit.safetensors", sha256=sha(tmp_path / "exit.safetensors")
        ),
    )
    write_manifest(tmp_path, manifest)
    before = {p.name: sha(p) for p in tmp_path.iterdir()}
    restored = OptionModel.load(tmp_path, device="cpu", mode=mode)
    assert requested == [
        (
            restored.encoder.config.to_dict(),
            ("cpu", ".cache/huggingface", False),
            dict(model_id=MERGED["model_id"], revision=MERGED["revision"]),
        )
    ]
    assert restored.backbone == MERGED
    assert restored.serving_mode == mode
    assert restored.calibration == model.calibration
    assert before == {p.name: sha(p) for p in tmp_path.iterdir()}
    for name, value in model.trainable_state().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
