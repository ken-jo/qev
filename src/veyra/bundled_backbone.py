"""Offline, integrity-checked Qwen3.5 backbones for format-3 option checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


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
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or "\\" in name
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
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", {})
    if (
        config.get("model_type") != "qwen3_5"
        or text.get("num_hidden_layers") != block["depth"]
        or text.get("vocab_size") != block["vocabulary"]
    ):
        raise ValueError("bundled depth/vocabulary does not match Qwen3.5 config")
    index = root / "model.safetensors.index.json"
    weights = {n for n in actual if n.endswith(".safetensors")}
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if shards != weights:
            raise ValueError("bundled weight shard inventory mismatch")
    elif weights != {"model.safetensors"}:
        raise ValueError("bundled weights need model.safetensors or a complete shard index")
    return root


def load_bundle(directory, block, config, device):
    """Load only verified local bytes, rejecting incomplete or incompatible weights."""
    import torch
    from transformers import AutoProcessor, Qwen3_5Model

    from veyra.backbone import QwenEncoder

    root = verify_bundle(directory, block)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for CPU execution")
    processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    exported = json.loads((root / "tokenizer.json").read_text())
    if json.loads(processor.tokenizer.backend_tokenizer.to_str()) != exported:
        raise ValueError("processor changed bundled tokenizer IDs or merge rules")
    if len(processor.tokenizer) != block["vocabulary"]:
        raise ValueError("bundled tokenizer vocabulary mismatch")
    processor.image_processor.size = {
        "shortest_edge": config.image_min_pixels,
        "longest_edge": config.image_max_pixels,
    }
    model, info = Qwen3_5Model.from_pretrained(
        root,
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
    model.eval().requires_grad_(False)
    return QwenEncoder(model, processor, config)
