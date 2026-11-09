"""Real tiny Qwen3.5 checkpoint loading, integrity failures and legacy regression."""

import hashlib
import json
import struct

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


def rehash(path, manifest, name):
    manifest["backbone"]["files"][name] = sha(path / "backbone" / name)
    write_manifest(path, manifest)


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


def test_transformers_full_config_serialization_loads(bundled):
    from transformers import Qwen3_5Config

    path, manifest, original = bundled
    config = path / "backbone/config.json"
    values = json.loads(config.read_text())
    serialized = Qwen3_5Config.from_dict(values).to_json_string(use_diff=False)
    full = json.loads(serialized)
    for scope in (full, full["text_config"], full["vision_config"]):
        assert scope["_name_or_path"] == ""
    config.write_text(serialized)
    rehash(path, manifest, "config.json")
    restored = OptionModel.load(path, device="cpu")
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("scope", [None, "text_config", "vision_config"])
@pytest.mark.parametrize("provenance", ["example/Qwen3.5-tiny", "", {"file": "../unused"}])
def test_model_config_provenance_is_inert(bundled, scope, provenance):
    path, manifest, _ = bundled
    config = path / "backbone/config.json"
    values = json.loads(config.read_text())
    target = values if scope is None else values[scope]
    target["_name_or_path"] = provenance
    config.write_text(json.dumps(values))
    rehash(path, manifest, "config.json")
    assert OptionModel.load(path, device="cpu").serving_mode == "accurate"


def test_textual_sep_token_metadata_preserves_probabilities(bundled):
    path, manifest, _ = bundled
    baseline = OptionModel.load(path, device="cpu")
    config = path / "backbone/config.json"
    values = json.loads(config.read_text())
    values["sep_token_id"] = 259
    config.write_text(json.dumps(values))
    rehash(path, manifest, "config.json")
    restored = OptionModel.load(path, device="cpu")
    assert restored.encoder.model.config.sep_token_id == 259
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
    with torch.inference_mode():
        expected = baseline.predict(request)["answers"]["q"]["probabilities"]
        actual = restored.predict(request)["answers"]["q"]["probabilities"]
    assert actual == expected


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


@pytest.mark.parametrize(
    "reference",
    [
        "../tokenizer.5.0.0.json",
        "/tmp/tokenizer.5.0.0.json",
        "tokenizer.5.0.0.json",
        "./tokenizer.json",
        r"..\tokenizer.json",
    ],
)
@pytest.mark.parametrize("filename", ["tokenizer_config.json", "processor_config.json"])
def test_config_file_indirections_are_rejected_before_processor(
    bundled, monkeypatch, filename, reference
):
    path, manifest, _ = bundled
    config = path / "backbone" / filename
    values = json.loads(config.read_text())
    if filename == "tokenizer_config.json":
        values["fast_tokenizer_files"] = [reference]
    else:
        values["image_processor"]["vocab_file"] = reference
    config.write_text(json.dumps(values))
    rehash(path, manifest, filename)
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *_a, **_k: pytest.fail("processor opened an unverified reference"),
    )
    with pytest.raises(ValueError, match="bundled file reference"):
        OptionModel.load(path, device="cpu")


@pytest.mark.parametrize("reference", [{"../tokenizer.5.0.0.json": "tokenizer.json"}, None, [42]])
def test_malformed_fast_tokenizer_reference_is_rejected(bundled, monkeypatch, reference):
    path, manifest, _ = bundled
    config = path / "backbone/tokenizer_config.json"
    values = json.loads(config.read_text())
    values["fast_tokenizer_files"] = reference
    config.write_text(json.dumps(values))
    rehash(path, manifest, "tokenizer_config.json")
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *_a, **_k: pytest.fail("processor opened a malformed reference"),
    )
    with pytest.raises(ValueError, match="bundled file reference"):
        OptionModel.load(path, device="cpu")


@pytest.mark.parametrize("scope", [None, "text_config", "vision_config"])
@pytest.mark.parametrize("reference", ["../audio-tokenizer.json", "", "unlisted.json"])
def test_component_name_or_path_is_validated(bundled, monkeypatch, scope, reference):
    path, manifest, _ = bundled
    config = path / "backbone/config.json"
    values = json.loads(config.read_text())
    target = values if scope is None else values[scope]
    target["_name_or_path"] = "example/Qwen3.5-tiny"
    target["audio_tokenizer_name_or_path"] = reference
    config.write_text(json.dumps(values))
    rehash(path, manifest, "config.json")
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *_a, **_k: pytest.fail("processor opened an unverified component reference"),
    )
    with pytest.raises(ValueError, match="bundled file reference"):
        OptionModel.load(path, device="cpu")


@pytest.mark.parametrize("extra", ["mtp.extra.weight", "rotary_emb.inv_freq"])
def test_hashed_transformers_ignored_weights_are_rejected(bundled, extra):
    from safetensors.torch import load_file, save_file

    path, manifest, _ = bundled
    weights = path / "backbone/model.safetensors"
    state = load_file(weights)
    state[extra] = torch.zeros(1)
    save_file(state, weights)
    rehash(path, manifest, "model.safetensors")
    with pytest.raises(ValueError, match="weight keys or shapes mismatch"):
        OptionModel.load(path, device="cpu")


def test_hashed_weight_shape_mismatch_is_rejected_before_model_load(bundled, monkeypatch):
    from safetensors.torch import load_file, save_file

    path, manifest, _ = bundled
    weights = path / "backbone/model.safetensors"
    state = load_file(weights)
    key = next(iter(state))
    state[key] = state[key].flatten()[:1]
    save_file(state, weights)
    rehash(path, manifest, "model.safetensors")
    monkeypatch.setattr(
        "transformers.Qwen3_5Model.from_pretrained",
        lambda *_a, **_k: pytest.fail("model loaded before header validation"),
    )
    with pytest.raises(ValueError, match="weight keys or shapes mismatch"):
        OptionModel.load(path, device="cpu")


@pytest.mark.parametrize(
    "fault", [None, "duplicate", "duplicate_index", "wrong_shard", "missing_key", "extra_key"]
)
def test_sharded_bundle_checks_raw_keys_and_index(bundled, fault):
    from safetensors.torch import load_file, save_file

    path, manifest, original = bundled
    root = path / "backbone"
    state = load_file(root / "model.safetensors")
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    items = list(state.items())
    shards = [dict(items[::2]), dict(items[1::2])]
    mapping = {key: name for name, shard in zip(names, shards) for key in shard}
    if fault == "duplicate":
        key = next(iter(shards[0]))
        shards[1][key] = shards[0][key]
    elif fault == "wrong_shard":
        key = next(iter(shards[0]))
        mapping[key] = names[1]
    elif fault == "missing_key":
        mapping.pop(next(iter(shards[0])))
    elif fault == "extra_key":
        mapping["mtp.not_in_any_shard"] = names[0]
    for name, shard in zip(names, shards):
        save_file(shard, root / name)
        rehash(path, manifest, name)
    (root / "model.safetensors").unlink()
    del manifest["backbone"]["files"]["model.safetensors"]
    index = "model.safetensors.index.json"
    (root / index).write_text(json.dumps({"metadata": {}, "weight_map": mapping}))
    if fault == "duplicate_index":
        key, name = next(iter(mapping.items()))
        text = (root / index).read_text()
        text = text.replace(
            '"weight_map": {', f'"weight_map": {{{json.dumps(key)}: {json.dumps(name)},'
        )
        (root / index).write_text(text)
    rehash(path, manifest, index)
    if fault:
        with pytest.raises(ValueError, match="duplicate|index mapping mismatch"):
            OptionModel.load(path, device="cpu")
    else:
        restored = OptionModel.load(path, device="cpu")
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_duplicate_raw_safetensors_header_key_is_rejected(bundled):
    path, manifest, _ = bundled
    weights = path / "backbone/model.safetensors"
    data = weights.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    header = data[8 : 8 + length].decode()
    key, value = next((k, v) for k, v in json.loads(header).items() if k != "__metadata__")
    header = "{" + json.dumps(key) + ":" + json.dumps(value) + "," + header[1:].rstrip()
    header += " " * (-len(header) % 8)
    weights.write_bytes(struct.pack("<Q", len(header)) + header.encode() + data[8 + length :])
    rehash(path, manifest, "model.safetensors")
    with pytest.raises(ValueError, match="duplicate keys"):
        OptionModel.load(path, device="cpu")


def test_verified_versioned_tokenizer_symlink_loads_and_hashes_target(bundled, tmp_path):
    path, manifest, _ = bundled
    root = path / "backbone"
    name = "tokenizer.5.0.0.json"
    blob = tmp_path / "tokenizer-blob"
    blob.write_bytes((root / "tokenizer.json").read_bytes() + b"\n")
    (root / name).symlink_to(blob)
    rehash(path, manifest, name)
    config = root / "tokenizer_config.json"
    values = json.loads(config.read_text())
    values["fast_tokenizer_files"] = [name]
    config.write_text(json.dumps(values))
    rehash(path, manifest, "tokenizer_config.json")
    assert OptionModel.load(path, device="cpu").serving_mode == "accurate"
    blob.write_bytes(blob.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        OptionModel.load(path, device="cpu")


@pytest.mark.parametrize(
    "field", ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]
)
@pytest.mark.parametrize("invalid", ["other_token", "out_of_bounds", "boolean"])
def test_hashed_multimodal_token_mismatch_is_rejected(bundled, field, invalid):
    path, manifest, _ = bundled
    config = path / "backbone/config.json"
    values = json.loads(config.read_text())
    values[field] = {
        "other_token": (values[field] + 1) % manifest["backbone"]["vocabulary"],
        "out_of_bounds": manifest["backbone"]["vocabulary"],
        "boolean": True,
    }[invalid]
    config.write_text(json.dumps(values))
    rehash(path, manifest, "config.json")
    with pytest.raises(ValueError, match="multimodal token"):
        OptionModel.load(path, device="cpu")


def test_loaded_bundle_processes_image_and_runs_vision_forward(bundled):
    from PIL import Image

    path, _, _ = bundled
    restored = OptionModel.load(path, device="cpu")
    encoder = restored.encoder
    processor = encoder.processor
    processor.image_processor.size = {"shortest_edge": 1024, "longest_edge": 1024}
    inputs = processor(
        text=["<|vision_start|><|image_pad|><|vision_end|> Test"],
        images=[Image.new("RGB", (32, 32), "red")],
        return_tensors="pt",
    )
    image_id = encoder.model.config.image_token_id
    expected = inputs.image_grid_thw.prod().item() // processor.image_processor.merge_size**2
    assert (inputs.input_ids == image_id).sum().item() == expected > 0
    with torch.inference_mode():
        hidden = encoder.model(**inputs, use_cache=False).last_hidden_state
    assert hidden.shape[:2] == inputs.input_ids.shape
    assert torch.isfinite(hidden).all()


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
