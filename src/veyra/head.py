"""A permutation-equivariant scoring head over a variable candidate set."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HeadConfig:
    input_size: int = 2048
    hidden_size: int = 256
    layers: int = 2
    attention_heads: int = 4

    def __post_init__(self) -> None:
        if min(self.input_size, self.hidden_size, self.layers, self.attention_heads) < 1:
            raise ValueError("head dimensions must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionHead(nn.Module):
    def __init__(self, config: HeadConfig = HeadConfig()) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Sequential(
            nn.LayerNorm(config.input_size),
            nn.Linear(config.input_size, config.hidden_size),
            nn.GELU(),
        )
        self.type_embedding = nn.Embedding(3, config.hidden_size)
        self.context_project = nn.Linear(config.input_size, config.hidden_size)
        self.level_embedding = nn.Linear(1, config.hidden_size, bias=False)
        layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.attention_heads,
            dim_feedforward=config.hidden_size * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.compare = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        # TransformerEncoder clones layer weights. Initialize each clone independently.
        for block in self.compare.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(block.linear1.weight)
            nn.init.xavier_uniform_(block.linear2.weight)
        self.scorer = nn.Sequential(
            nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, 1)
        )

    def forward(
        self,
        features: Tensor,
        valid: Tensor,
        question_types: Tensor,
        levels: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.input_size:
            raise ValueError("features must have shape [batch, candidates, input_size]")
        batch, count = features.shape[:2]
        if valid.shape != (batch, count) or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [batch, candidates] mask")
        if levels.shape != valid.shape or question_types.shape != (batch,):
            raise ValueError("question type or ordinal level shape mismatch")
        if question_types.dtype != torch.long or torch.any(
            (question_types < 0) | (question_types > 2)
        ):
            raise ValueError("question_types must be integer type indices 0, 1, or 2")
        if torch.any(valid.sum(-1) < 2):
            raise ValueError("each question must have at least two valid candidates")
        if not torch.isfinite(features[valid]).all() or not torch.isfinite(levels[valid]).all():
            raise ValueError("valid candidate features and levels must be finite")
        clean = features.float().masked_fill(~valid.unsqueeze(-1), 0)
        ordinal = levels.float().masked_fill(~valid, 0)
        ordinal = ordinal * (question_types == 1).unsqueeze(-1)
        values = (
            self.project(clean)
            + self.type_embedding(question_types).unsqueeze(1)
            + self.level_embedding(ordinal.unsqueeze(-1))
        )
        if context is not None:
            if (
                context.shape != (batch, self.config.input_size)
                or not torch.isfinite(context).all()
            ):
                raise ValueError("context must be finite [batch, input_size]")
            values = values + self.context_project(context.float()).unsqueeze(1)
        compared = self.compare(values, src_key_padding_mask=~valid)
        return self.scorer(compared).squeeze(-1).masked_fill(~valid, -torch.inf)
