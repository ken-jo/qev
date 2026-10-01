"""Pinned Qwen3.5-2B multimodal encoding with a single prefill per request."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import Tensor, nn

from veyra.candidates import SYSTEM_PROMPT, type_index
from veyra.constants import MODEL_ID, MODEL_REVISION
from veyra.packing import CANDIDATE_MARKER, QUESTION_MARKER, marker_positions, pack_request
from veyra.schema import DecisionRequest


@dataclass(frozen=True)
class EncoderConfig:
    max_total_tokens: int = 2048
    image_max_pixels: int = 262144
    image_min_pixels: int = 65536
    encoding: str = "packed-v1"

    def __post_init__(self) -> None:
        if not 1 <= self.max_total_tokens <= 8192:
            raise ValueError("max_total_tokens must be in [1, 8192]")
        if not 1024 <= self.image_min_pixels <= self.image_max_pixels <= 1048576:
            raise ValueError("invalid image pixel budget")
        if self.encoding != "packed-v1":
            raise ValueError("unsupported encoding format")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EncodedQuestion:
    features: Tensor
    context: Tensor
    levels: Tensor
    type_id: int


@dataclass
class EncodedRequest:
    questions: dict[str, EncodedQuestion]
    input_tokens: int


class QwenEncoder(nn.Module):
    def __init__(self, model: nn.Module, processor, config: EncoderConfig) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config

    @classmethod
    def load(
        cls,
        config: EncoderConfig = EncoderConfig(),
        device: str = "cuda",
        cache_dir: str = ".cache/huggingface",
        local_files_only: bool = False,
    ) -> QwenEncoder:
        from transformers import AutoProcessor, Qwen3_5Model

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for CPU execution")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        processor.image_processor.size = {
            "shortest_edge": config.image_min_pixels,
            "longest_edge": config.image_max_pixels,
        }
        model = Qwen3_5Model.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=False,
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            device_map={"": device},
            attn_implementation="sdpa",
        )
        model.eval().requires_grad_(False)
        return cls(model, processor, config)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, request: DecisionRequest, image_root: Path | None = None) -> EncodedRequest:
        packed = pack_request(request)
        image = None
        if request.state.images:
            path = Path(request.state.images[0].path)
            if image_root is not None:
                root = image_root.resolve()
                path = (root / path).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("image path is outside the configured image root")
            if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("image must be a local file no larger than 20 MiB")
            with Image.open(path) as source:
                if source.width * source.height > 40_000_000:
                    raise ValueError("source image exceeds the 40 megapixel decode limit")
                image = ImageOps.exif_transpose(source).convert("RGB")
        content = ([{"type": "image"}] if image is not None else []) + [
            {"type": "text", "text": packed.text}
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        try:
            inputs = self.processor(
                text=[prompt],
                images=[image] if image is not None else None,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
        finally:
            if image is not None:
                image.close()
        ids = inputs["input_ids"][0].tolist()
        if len(ids) > self.config.max_total_tokens:
            raise ValueError(
                f"request has {len(ids)} tokens, exceeding {self.config.max_total_tokens}; "
                "reduce input size explicitly"
            )
        positions = {}
        for name, marker, count in (
            ("candidates", CANDIDATE_MARKER, packed.candidate_count),
            ("questions", QUESTION_MARKER, packed.question_count),
        ):
            marker_ids = self.processor.tokenizer.encode(marker, add_special_tokens=False)
            positions[name] = marker_positions(ids, marker_ids, count)
        inputs = inputs.to(self.device)
        grad_context = (
            contextlib.nullcontext()
            if any(p.requires_grad for p in self.model.parameters())
            else torch.no_grad()
        )
        with grad_context:
            hidden = self.model(**inputs, use_cache=False).last_hidden_state[0]
            candidate_states = hidden[positions["candidates"]].float()
            question_states = hidden[positions["questions"]].float()
        encoded = {}
        for name, layout in packed.layouts.items():
            encoded[name] = EncodedQuestion(
                features=candidate_states[list(layout.candidate_indices)],
                context=question_states[layout.context_index],
                levels=torch.tensor([c.level for c in layout.candidates], device=self.device),
                type_id=type_index(layout.question),
            )
        return EncodedRequest(encoded, len(ids))
