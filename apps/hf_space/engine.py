"""Public demo adapter around the released, unchanged decision runtime."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

from qwen3_5_classification import DecisionRequest, QwenClassification

MODEL_ID = "ken-jo/qwen3.5-classification"
MODEL_REVISION = "0d2d13ffb3c392071ea00b6bdb903c4b84dff48d"
WEIGHTS_SHA = "84b57aeeb987f73416ac0f796150957d1c5beb1ed373733053e46438f77a8fee"
MANIFEST_SHA = "d83f9910196c6e801658ca3816c3d2bd4a849ba3da6ff059a0edf7c6028e898d"
MAX_IMAGE_PIXELS = 16_000_000
RESOLUTIONS = (64, 128, 192, 256, 384, 512, 768, 1024)


def build_request(text, question, kind, criteria, has_image):
    if len(text) > 6000 or len(question) > 1000 or len(criteria) > 6000:
        raise ValueError("Please shorten the evidence, question or candidate descriptions.")
    lines = [line.strip() for line in criteria.splitlines() if line.strip()]
    if kind == "score":
        choices = lines
    elif kind in {"choice", "noul"}:
        choices = {}
        for line in lines:
            if "|" not in line:
                raise ValueError("Use one candidate per line: label | description.")
            label, description = (part.strip() for part in line.split("|", 1))
            if label in choices:
                raise ValueError("Candidate labels must be unique.")
            choices[label] = description
        if kind == "noul" and set(choices) != {"false", "true"}:
            raise ValueError("A noul question needs exactly the labels false and true.")
    else:
        raise ValueError("Choose choice, score or noul.")
    return DecisionRequest.model_validate(
        {
            "state": {"text": text, "images": [{"path": "image.png"}] if has_image else []},
            "questions": {
                "decision": {"type": kind, "instructions": question, "criteria": choices}
            },
        }
    )


def prepare_image(image, resolution):
    if image is None:
        return None, "No image"
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Use an image of at most 16 megapixels.")
    if resolution != "Original" and int(resolution) not in RESOLUTIONS:
        raise ValueError("Choose one of the available image resolutions.")
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    source_size = prepared.size
    if resolution != "Original":
        edge = int(resolution)
        prepared.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    return prepared, f"{source_size[0]} x {source_size[1]} -> {prepared.width} x {prepared.height}"


class DemoEngine:
    def __init__(self, device="cpu", checkpoint=None, cache_dir=None):
        if device not in {"cpu", "cuda"}:
            raise ValueError("Unsupported demo device")
        self.device = device
        self.checkpoint = checkpoint
        self.cache_dir = str(cache_dir or os.environ.get("QEV_CACHE_DIR", ".cache/huggingface"))
        self.model = None

    def load(self):
        from veyra.constants import MODEL_ID as BASE_ID
        from veyra.constants import MODEL_REVISION as BASE_REVISION

        torch.set_num_threads(2)
        if self.checkpoint is None:
            self.checkpoint = snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                allow_patterns=["head.safetensors", "manifest.json"],
                cache_dir=self.cache_dir,
                token=False,
            )
        folder = Path(self.checkpoint)
        for name, expected in (("head.safetensors", WEIGHTS_SHA), ("manifest.json", MANIFEST_SHA)):
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError("The released checkpoint failed its integrity check.")
        snapshot_download(
            BASE_ID,
            revision=BASE_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
            cache_dir=self.cache_dir,
            local_files_only=os.environ.get("QEV_OFFLINE") == "1",
            token=False,
        )
        self.model = QwenClassification.load(
            folder,
            device=self.device,
            cache_dir=self.cache_dir,
            local_files_only=True,
            merge=True,
        )
        return self

    def predict(self, text, image, question, kind, criteria, resolution="Original"):
        if self.model is None:
            raise RuntimeError("The model is not ready yet. Please try again shortly.")
        request = build_request(text, question, kind, criteria, image is not None)
        prepared, geometry = prepare_image(image, resolution)
        try:
            with tempfile.TemporaryDirectory(prefix="qwen-classification-") as temporary:
                if prepared is not None:
                    prepared.save(Path(temporary) / "image.png")
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = self.model.predict(request, image_root=Path(temporary))
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
        finally:
            if prepared is not None:
                prepared.close()
        answer = result["answers"]["decision"]
        probabilities = answer["probabilities"]
        top = max(probabilities, key=probabilities.get)
        if kind == "score":
            expected = sum(float(key) * value for key, value in probabilities.items())
            decision = f"Expected score: {expected:.3f} (levels start at 0)"
        elif kind == "noul":
            decision = f"Probability true: {probabilities['true']:.1%}"
        else:
            decision = f"Top candidate: {top} ({probabilities[top]:.1%})"
        status = "ABSTAINED - review the evidence" if answer["abstained"] else "Answer accepted"
        summary = (
            f"{status}\n{decision}\n"
            f"Model computation: {elapsed:.0f} ms on {self.device.upper()}\n"
            f"Image: {geometry}\nQueue, upload and network time are excluded."
        )
        result["demo"] = {"device": self.device, "model_ms": round(elapsed, 2), "image": geometry}
        # Keep schema paths as JSON text: Gradio clients otherwise interpret them as files.
        return summary, probabilities, json.dumps(request.model_dump(), indent=2), result
