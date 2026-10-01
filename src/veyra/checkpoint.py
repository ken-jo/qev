"""Versioned head checkpoints referencing the exact shared backbone revision."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from safetensors.torch import load_file, save_file

from veyra.backbone import EncoderConfig
from veyra.constants import MODEL_ID, MODEL_REVISION
from veyra.head import DecisionHead, HeadConfig
from veyra.probability import Calibration


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(
    directory: str | Path,
    head: DecisionHead,
    encoder_config: EncoderConfig,
    calibration: Calibration,
    training: dict,
) -> dict:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("checkpoint directory must be empty; use a new checkpoint path")
    weights = path / "head.safetensors"
    temporary = path / "head.safetensors.tmp"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()},
        temporary,
    )
    temporary.replace(weights)
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "backbone": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "head": head.config.to_dict(),
        "encoder": encoder_config.to_dict(),
        "calibration": calibration.to_dict(),
        "training": training,
        "weights_sha256": sha256_file(weights),
    }
    temporary_manifest = path / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(path / "manifest.json")
    return manifest


def load_head(directory: str | Path, require_trained: bool = True) -> tuple:
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if manifest.get("backbone") != {"model_id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("checkpoint backbone does not match the pinned Qwen3.5-2B revision")
    if require_trained and (
        manifest.get("training", {}).get("completed") is not True
        or manifest["training"].get("optimizer_steps", 0) < 1
    ):
        raise ValueError("checkpoint does not contain a completed head training run")
    weights = path / "head.safetensors"
    if sha256_file(weights) != manifest.get("weights_sha256"):
        raise ValueError("checkpoint weight checksum mismatch")
    head = DecisionHead(HeadConfig(**manifest["head"]))
    head.load_state_dict(load_file(weights), strict=True)
    head.eval()
    return (
        head,
        EncoderConfig(**manifest["encoder"]),
        Calibration.from_dict(manifest["calibration"]),
        manifest,
    )
