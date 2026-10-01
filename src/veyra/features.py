"""Resumable frozen-backbone feature extraction with input and code fingerprints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from veyra.backbone import EncoderConfig, QwenEncoder
from veyra.candidates import candidates_for
from veyra.checkpoint import sha256_file
from veyra.constants import MODEL_ID, MODEL_REVISION
from veyra.data import TrainingRecord, read_records


@dataclass
class FeatureSample:
    features: Tensor
    context: Tensor
    levels: Tensor
    targets: Tensor
    type_id: int
    record_id: str
    group_id: str
    split: str
    family: str
    language: str
    question_id: str
    tags: list[str]


def fingerprint(records_path: Path, config: EncoderConfig) -> tuple[str, dict]:
    package = Path(__file__).parent
    source_hash = hashlib.sha256()
    for name in ("backbone.py", "packing.py", "candidates.py", "features.py"):
        source_hash.update(name.encode())
        source_hash.update((package / name).read_bytes())
    specification = {
        "format_version": 1,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "encoder": config.to_dict(),
        "dataset_sha256": sha256_file(records_path),
        "encoding_source_sha256": source_hash.hexdigest(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "transformers", "pillow")
        },
    }
    digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    return digest, specification


def chunk_path(cache: Path, record: TrainingRecord) -> Path:
    return cache / "chunks" / (hashlib.sha256(record.id.encode()).hexdigest() + ".safetensors")


@torch.inference_mode()
def extract_features(
    records_path: Path,
    output: Path,
    device: str = "cuda",
    cache_dir: str = ".cache/huggingface",
    config: EncoderConfig = EncoderConfig(),
) -> dict:
    records = read_records(records_path)
    digest, specification = fingerprint(records_path, config)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != digest:
            raise ValueError("feature cache belongs to different data, code, or encoder settings")
    manifest = {"fingerprint": digest, "specification": specification, "complete": False}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "chunks").mkdir(exist_ok=True)
    encoder = QwenEncoder.load(config, device, cache_dir, local_files_only=True).eval()
    completed = 0
    tokens = []
    for index, record in enumerate(records):
        target = chunk_path(output, record)
        if target.exists():
            with safe_open(target, framework="pt") as existing:
                metadata = existing.metadata()
                if metadata.get("fingerprint") != digest or metadata.get("record_id") != record.id:
                    raise ValueError("feature chunk metadata mismatch")
            completed += 1
            continue
        encoded = encoder(record.request, image_root=records_path.parent)
        tensors = {}
        for name, question in encoded.questions.items():
            # Training data producers control question IDs; hashes avoid tensor-key collisions.
            key = hashlib.sha256(name.encode()).hexdigest()
            tensors[key + ".features"] = question.features.cpu().half().contiguous()
            tensors[key + ".context"] = question.context.cpu().half().contiguous()
            tensors[key + ".levels"] = question.levels.cpu().float().contiguous()
            if not all(
                torch.isfinite(tensor).all()
                for tensor in (tensors[key + ".features"], tensors[key + ".context"])
            ):
                raise ValueError("nonfinite frozen features cannot be cached")
        temporary = target.with_suffix(".tmp")
        save_file(
            tensors,
            temporary,
            metadata={
                "fingerprint": digest,
                "record_id": record.id,
                "input_tokens": str(encoded.input_tokens),
            },
        )
        temporary.replace(target)
        completed += 1
        tokens.append(encoded.input_tokens)
        if (index + 1) % 50 == 0 or index == len(records) - 1:
            print(f"features: {completed}/{len(records)} records", flush=True)
    manifest.update(
        {"complete": True, "records": completed, "max_tokens_this_run": max(tokens, default=0)}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_features(records_path: Path, cache: Path) -> tuple[list[FeatureSample], dict]:
    records = read_records(records_path)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    config = EncoderConfig(**manifest["specification"]["encoder"])
    digest, _ = fingerprint(records_path, config)
    if not manifest.get("complete") or manifest.get("fingerprint") != digest:
        raise ValueError("feature cache is incomplete or stale")
    samples = []
    for record in records:
        path = chunk_path(cache, record)
        with safe_open(path, framework="pt") as reader:
            metadata = reader.metadata()
            if metadata.get("fingerprint") != digest or metadata.get("record_id") != record.id:
                raise ValueError("feature chunk does not match its record")
        tensors = load_file(path)
        for name, question in record.request.questions.items():
            key = hashlib.sha256(name.encode()).hexdigest()
            candidates = candidates_for(question)
            samples.append(
                FeatureSample(
                    features=tensors[key + ".features"].float(),
                    context=tensors[key + ".context"].float(),
                    levels=tensors[key + ".levels"],
                    targets=torch.tensor([record.targets[name][c.key] for c in candidates]),
                    type_id=("choice", "score", "noul").index(question.type),
                    record_id=record.id,
                    group_id=record.group_id,
                    split=record.split,
                    family=record.family,
                    language=record.language,
                    question_id=name,
                    tags=record.tags,
                )
            )
    return samples, manifest


def collate(samples: list[FeatureSample], device: str = "cpu") -> dict[str, Tensor]:
    if not samples:
        raise ValueError("empty feature batch")
    count = max(sample.features.shape[0] for sample in samples)
    width = samples[0].features.shape[1]
    features = torch.zeros(len(samples), count, width)
    levels = torch.zeros(len(samples), count)
    targets = torch.zeros(len(samples), count)
    valid = torch.zeros(len(samples), count, dtype=torch.bool)
    for i, sample in enumerate(samples):
        n = sample.features.shape[0]
        features[i, :n] = sample.features
        levels[i, :n] = sample.levels
        targets[i, :n] = sample.targets
        valid[i, :n] = True
    return {
        name: tensor.to(device)
        for name, tensor in {
            "features": features,
            "levels": levels,
            "targets": targets,
            "valid": valid,
            "question_types": torch.tensor(
                [sample.type_id for sample in samples], dtype=torch.long
            ),
            "context": torch.stack([sample.context for sample in samples]),
        }.items()
    }
