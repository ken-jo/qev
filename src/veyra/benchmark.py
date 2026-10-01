"""Whole-request timing with resident weights and no feature/cache reuse."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from veyra.backbone import EncoderConfig, QwenEncoder
from veyra.constants import MODEL_ID, MODEL_REVISION
from veyra.head import DecisionHead
from veyra.model import VeyraModel
from veyra.probability import typed_answer
from veyra.schema import DecisionRequest


def write_fixture(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / "shapes.png"
    with Image.new("RGB", (384, 256), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((35, 75, 135, 175), fill="blue")
        draw.ellipse((230, 75, 330, 175), fill="red")
        image.save(image_path)
    request = {
        "state": {
            "text": "Inspect the pictured objects.",
            "images": [{"path": str(image_path.resolve())}],
        },
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "What color is the square?",
                "criteria": {"a": "blue", "b": "red", "c": "green", "d": "yellow"},
            }
        },
    }
    request_path = directory / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    return request_path


@torch.inference_mode()
def profile(
    request_path: Path,
    output_path: Path,
    *,
    device: str = "cuda",
    cache_dir: str = ".cache/huggingface",
    checkpoint: str | None = None,
    warmup: int = 3,
    iterations: int = 20,
) -> dict:
    if warmup < 1 or iterations < 5:
        raise ValueError("profile requires at least one warmup and five measured requests")
    text = request_path.read_text(encoding="utf-8")
    request = DecisionRequest.from_json(text)
    print("Loading pinned Qwen3.5-2B...", flush=True)
    if checkpoint:
        model = VeyraModel.load(checkpoint, device, cache_dir, local_files_only=True)
    else:
        model = VeyraModel(
            QwenEncoder.load(EncoderConfig(), device, cache_dir, local_files_only=True),
            DecisionHead(),
        ).eval()
    print("Model loaded; running complete requests without feature caching...", flush=True)

    def execute() -> int:
        current = DecisionRequest.from_json(text)
        logits, tokens = model(current)
        answers = {
            name: typed_answer(q, logits[name], model.calibration)
            for name, q in current.questions.items()
        }
        json.dumps(answers)
        return tokens

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    for _ in range(warmup):
        execute()
    synchronize()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    milliseconds = []
    for i in range(iterations):
        synchronize()
        start = time.perf_counter()
        tokens = execute()
        synchronize()
        milliseconds.append((time.perf_counter() - start) * 1000)
        print(f"request {i + 1}/{iterations}: {milliseconds[-1]:.1f} ms", flush=True)
    report = {
        "backbone": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "encoding": model.encoder.config.to_dict(),
        "head_trained": model.trained,
        "scope": (
            "JSON parsing, image decode, processing, one Qwen prefill, "
            "head, typed JSON serialization"
        ),
        "excludes": ["model loading", "HTTP transport"],
        "feature_cache": False,
        "fixture_repeated": True,
        "input_tokens": tokens,
        "questions": len(request.questions),
        "warmup_requests": warmup,
        "measured_requests": iterations,
        "latency_ms": {
            "p50": float(np.percentile(milliseconds, 50)),
            "p95": float(np.percentile(milliseconds, 95)),
            "min": min(milliseconds),
            "max": max(milliseconds),
            "samples": milliseconds,
        },
        "target_ms": 300,
        "p95_target_met": bool(np.percentile(milliseconds, 95) <= 300),
        "device": device,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if device.startswith("cuda")
        else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
        if device.startswith("cuda")
        else None,
        "torch_version": torch.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "latency_ms"}, indent=2), flush=True)
    return report
