"""Public demo adapter around the released, unchanged decision runtime."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from qev import DecisionRequest
from qev.runtime import load_model, resolve_cache, resolve_device

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
    def __init__(self, device="auto", checkpoint=None, cache_dir=None, offline=False):
        self.device = resolve_device(device)
        self.checkpoint = checkpoint
        self.cache_dir = resolve_cache(cache_dir)
        self.offline = offline or os.environ.get("QEV_OFFLINE") == "1"
        self.model = None

    def load(self, progress=None):
        torch.set_num_threads(4)
        self.model = load_model(
            self.checkpoint,
            device=self.device,
            cache_dir=self.cache_dir,
            offline=self.offline,
            progress=progress,
        )
        return self

    def predict(self, text, image, question, kind, criteria, resolution="Original"):
        if self.model is None:
            raise RuntimeError("The model is not ready yet. Please try again shortly.")
        request = build_request(text, question, kind, criteria, image is not None)
        prepared, geometry = prepare_image(image, resolution)
        try:
            with tempfile.TemporaryDirectory(prefix="qev-") as temporary:
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
