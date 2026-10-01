"""The shared Qwen backbone and typed decision head form one PyTorch module."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import torch
from torch import nn

from veyra.backbone import QwenEncoder
from veyra.checkpoint import load_head
from veyra.constants import MODEL_ID, MODEL_REVISION
from veyra.head import DecisionHead
from veyra.probability import Calibration, typed_answer
from veyra.schema import DecisionRequest


class VeyraModel(nn.Module):
    def __init__(
        self,
        encoder: QwenEncoder,
        head: DecisionHead,
        calibration: Calibration = Calibration(),
        trained: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head.to(encoder.device)
        self.calibration = calibration
        self.trained = trained
        self._request_lock = threading.RLock()

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        device: str = "cuda",
        cache_dir: str = ".cache/huggingface",
        local_files_only: bool = False,
    ) -> VeyraModel:
        manifest = json.loads((Path(checkpoint) / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("architecture") == "option_readout":
            from veyra.option_model import OptionModel

            return OptionModel.load(checkpoint, device, cache_dir, local_files_only)
        head, config, calibration, _ = load_head(checkpoint)
        encoder = QwenEncoder.load(config, device, cache_dir, local_files_only)
        return cls(encoder, head, calibration, trained=True).eval()

    def forward(self, request: DecisionRequest, image_root: Path | None = None) -> tuple[dict, int]:
        # Qwen stores mutable RoPE state even without generation caches.
        with self._request_lock:
            encoded = self.encoder(request, image_root=image_root)
            logits = {}
            for name, question in encoded.questions.items():
                count = question.features.shape[0]
                logits[name] = self.head(
                    question.features.unsqueeze(0),
                    torch.ones((1, count), dtype=torch.bool, device=question.features.device),
                    torch.tensor(
                        [question.type_id], dtype=torch.long, device=question.features.device
                    ),
                    question.levels.unsqueeze(0),
                    question.context.unsqueeze(0),
                )[0]
            return logits, encoded.input_tokens

    @torch.inference_mode()
    def predict(self, request: DecisionRequest, image_root: Path | None = None) -> dict:
        if not self.trained:
            raise RuntimeError("prediction requires a trained head checkpoint")
        self.eval()
        logits, tokens = self(request, image_root=image_root)
        return {
            "model": "veyra-qwen3.5-2b",
            "backbone": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
            "answers": {
                name: typed_answer(question, logits[name], self.calibration)
                for name, question in request.questions.items()
            },
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }
