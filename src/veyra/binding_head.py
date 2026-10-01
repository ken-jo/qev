"""A small condition-modulated residual over request-local option letters."""

import torch
from torch import nn


class ConditionedReadout(nn.Module):
    def __init__(self, rank=64, mode="condition", input_size=2048, dropout=0.1):
        super().__init__()
        if type(rank) is not int or not 1 <= rank <= 256:
            raise ValueError("binding rank must be an integer from 1 to 256")
        if mode not in {"condition", "uniform"}:
            raise ValueError("binding mode must be condition or uniform")
        self.mode = mode
        self.normalize = nn.LayerNorm(input_size)
        self.project = nn.Linear(input_size, rank)
        self.dropout = nn.Dropout(dropout)
        self.experts = nn.Linear(rank, 3 * 16, bias=False)
        # The untrained module preserves every original option logit exactly.
        nn.init.zeros_(self.experts.weight)

    def forward(self, states, condition_logits):
        if condition_logits.shape != (states.shape[0], 3):
            raise ValueError("expected three predicted condition logits per question")
        hidden = self.dropout(torch.nn.functional.gelu(self.project(self.normalize(states))))
        alternatives = self.experts(hidden).view(states.shape[0], 3, 16)
        weights = (
            condition_logits.softmax(-1)
            if self.mode == "condition"
            else torch.full_like(condition_logits, 1 / 3)
        )
        return (weights.unsqueeze(-1) * alternatives).sum(1)
