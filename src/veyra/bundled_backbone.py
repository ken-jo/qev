"""Offline, integrity-checked Qwen3.5 backbones for format-3 option checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path, PurePosixPath

# The pinned Qwen3.5 model and processor use these IDs for multimodal placeholders
# and boundaries. Other token metadata, such as sep_token_id, is textual.
_MULTIMODAL_FIELDS = (
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


def _relative_file_name(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and str(path) == name != "."


def _unique_weight_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("bundled weight inventory contains duplicate keys")
        result[key] = value
    return result


def _verify_config_references(root, files):
    """Check indirections before Transformers can select or open another file."""

    def reference(value, *, mapping=False):
        if value is None:
            return
        if isinstance(value, dict) and mapping:
            for item in value.values():
                reference(item)
        elif isinstance(value, list):
            for item in value:
                reference(item)
        elif not _relative_file_name(value) or value not in files:
            raise ValueError(f"invalid bundled file reference: {value!r}")

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                # These exact keys are inert save_pretrained provenance, including
                # in nested model configs. Do not inspect their metadata values.
                # Component *_name_or_path references still require verified files.
                if key in {"name_or_path", "_name_or_path"}:
                    continue
                if (
                    key
                    in {
                        "file",
                        "files",
                        "filename",
                        "filenames",
                        "path",
                        "paths",
                        "vocab_files_names",
                    }
                    or key.endswith(
                        ("_file", "_files", "_filename", "_filenames", "_path", "_paths")
                    )
                ):
                    if key == "fast_tokenizer_files" and (
                        not isinstance(item, list) or any(not isinstance(v, str) for v in item)
                    ):
                        raise ValueError("invalid bundled file reference: fast_tokenizer_files")
                    reference(item, mapping=key == "vocab_files_names")
                elif key == "subfolder" and item:
                    raise ValueError("invalid bundled file reference: subfolder")
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for name in files:
        if name.endswith("config.json"):
            walk(json.loads((root / name).read_text()))


def _verify_weight_state(root, files, expected):
    """Read every verified safetensors header, without Transformers' key filtering."""
    from safetensors import safe_open

    actual, locations = {}, {}
    for name in sorted(files):
        if not name.endswith(".safetensors"):
            continue
        # JSON duplicate members must not disappear when a header is decoded.
        # Safetensors caps headers at 100 MB; never read tensor payloads here.
        with (root / name).open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError("invalid bundled safetensors header")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > 100_000_000 or length > (root / name).stat().st_size - 8:
                raise ValueError("invalid bundled safetensors header length")
            json.loads(stream.read(length), object_pairs_hook=_unique_weight_keys)
        with safe_open(root / name, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key in actual:
                    raise ValueError(f"bundled weight keys or shapes mismatch: duplicate {key}")
                actual[key] = tuple(shard.get_slice(key).get_shape())
                locations[key] = name
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text(), object_pairs_hook=_unique_weight_keys).get(
            "weight_map"
        )
        if mapping != locations:
            raise ValueError("bundled weight index mapping mismatch")
    if actual != {key: tuple(value.shape) for key, value in expected.items()}:
        raise ValueError("bundled backbone weight keys or shapes mismatch")


def _verify_multimodal_tokens(model_config, processor, vocabulary):
    values = model_config if isinstance(model_config, dict) else model_config.to_dict()
    tokenizer = processor.tokenizer
    for field in _MULTIMODAL_FIELDS:
        value = values.get(field)
        token = getattr(processor, field.removesuffix("_id"), None)
        processor_id = getattr(processor, field, None)
        if (
            type(value) is not int
            or not 0 <= value < vocabulary
            or not isinstance(token, str)
            or token not in tokenizer.get_vocab()
            or tokenizer.convert_tokens_to_ids(token) != value
            or tokenizer.encode(token, add_special_tokens=False) != [value]
            or type(processor_id) is not int
            or processor_id != value
        ):
            raise ValueError(f"bundled multimodal token mismatch: {field}")


def bundled_identity(block):
    fields = {"bundled", "model_id", "revision", "depth", "vocabulary", "files"}
    if (
        not isinstance(block, dict)
        or set(block) != fields
        or block.get("bundled") is not True
        or not isinstance(block.get("model_id"), str)
        or not re.fullmatch(r"[\w.-]+/[\w.-]+", block["model_id"])
        or not isinstance(block.get("revision"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", block["revision"])
        or type(block.get("depth")) is not int
        or block["depth"] < 1
        or type(block.get("vocabulary")) is not int
        or block["vocabulary"] < 16
        or not isinstance(block.get("files"), dict)
        or not block["files"]
    ):
        raise ValueError("invalid bundled backbone identity")
    for name, digest in block["files"].items():
        if (
            not _relative_file_name(name)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("invalid bundled file name or SHA-256")
    return dict(block)


def verify_bundle(directory, block):
    """Verify the entire backbone directory before any Transformers loader runs.

    Hub snapshot file symlinks are supported: their bytes are hashed like regular files.
    Directory symlinks and unlisted files are rejected. The manifest is the trust root.
    """
    block = bundled_identity(block)
    root = Path(directory) / "backbone"
    if not root.is_dir() or root.is_symlink():
        raise ValueError("missing bundled backbone directory")
    paths = list(root.rglob("*"))
    if any(p.is_symlink() and p.is_dir() for p in paths):
        raise ValueError("bundled directory symlinks are forbidden")
    actual = {p.relative_to(root).as_posix() for p in paths if not p.is_dir()}
    if actual != set(block["files"]):
        raise ValueError("bundled file inventory mismatch (missing or unlisted file)")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json"}
    if not required <= actual or not any(n.endswith(".safetensors") for n in actual):
        raise ValueError("bundled backbone lacks config, weights, tokenizer or processor")
    if any("adapter" in n or n.endswith((".bin", ".py", ".pt", ".pth")) for n in actual):
        raise ValueError(
            "bundled backbone must contain safe dense weights, without adapters or code"
        )
    for name, expected in block["files"].items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing bundled file: {name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"bundled file checksum mismatch: {name}")
    _verify_config_references(root, actual)
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", {})
    if (
        config.get("model_type") != "qwen3_5"
        or text.get("num_hidden_layers") != block["depth"]
        or text.get("vocab_size") != block["vocabulary"]
    ):
        raise ValueError("bundled depth/vocabulary does not match Qwen3.5 config")
    for field in _MULTIMODAL_FIELDS:
        value = config.get(field)
        if type(value) is not int or not 0 <= value < block["vocabulary"]:
            raise ValueError(f"bundled multimodal token mismatch: {field}")
    index = root / "model.safetensors.index.json"
    weights = {n for n in actual if n.endswith(".safetensors")}
    if index.exists():
        mapping = json.loads(index.read_text()).get("weight_map")
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(
                not isinstance(key, str) or not _relative_file_name(name)
                for key, name in mapping.items()
            )
        ):
            raise ValueError("invalid bundled weight index mapping")
        shards = set(mapping.values())
        if shards != weights:
            raise ValueError("bundled weight shard inventory mismatch")
    elif weights != {"model.safetensors"}:
        raise ValueError("bundled weights need model.safetensors or a complete shard index")
    return root


def load_bundle(directory, block, config, device):
    """Load only verified local bytes, rejecting incomplete or incompatible weights."""
    import torch
    from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5Model

    from veyra.backbone import QwenEncoder

    root = verify_bundle(directory, block)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for CPU execution")
    model_values = json.loads((root / "config.json").read_text())
    processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    exported = json.loads((root / "tokenizer.json").read_text())
    if json.loads(processor.tokenizer.backend_tokenizer.to_str()) != exported:
        raise ValueError("processor changed bundled tokenizer IDs or merge rules")
    if len(processor.tokenizer) != block["vocabulary"]:
        raise ValueError("bundled tokenizer vocabulary mismatch")
    _verify_multimodal_tokens(model_values, processor, block["vocabulary"])
    model_config = Qwen3_5Config.from_dict(model_values)
    # A meta model provides the dense architecture's exact state without allocating
    # another copy of the backbone or trusting filtered loading diagnostics.
    with torch.device("meta"):
        expected = Qwen3_5Model(model_config)
    _verify_weight_state(root, block["files"], expected.state_dict())
    del expected
    processor.image_processor.size = {
        "shortest_edge": config.image_min_pixels,
        "longest_edge": config.image_max_pixels,
    }
    model, info = Qwen3_5Model.from_pretrained(
        root,
        config=model_config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map={"": device},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(
        info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("bundled backbone weight keys or shapes mismatch")
    language = model.language_model
    if (
        len(language.layers) != block["depth"]
        or language.embed_tokens.num_embeddings != block["vocabulary"]
    ):
        raise ValueError("loaded bundled depth/vocabulary mismatch")
    _verify_multimodal_tokens(model.config, processor, language.embed_tokens.num_embeddings)
    model.eval().requires_grad_(False)
    return QwenEncoder(model, processor, config)
