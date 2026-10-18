"""Pinned Qwen3.5-2B multimodal encoding with a single prefill per request."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from veyra.candidates import SYSTEM_PROMPT, type_index
from veyra.constants import MODEL_ID, MODEL_REVISION
from veyra.image_inputs import image_content, open_images
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
        # Draft against the processor's effective budget, which a caller may raise or lower.
        budget = self.processor.image_processor.size["longest_edge"]
        with open_images(request.state.images, image_root, max_pixels=budget) as images:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": image_content(len(images), packed.text)},
            ]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            inputs = self.processor(
                text=[prompt],
                images=images or None,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
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
