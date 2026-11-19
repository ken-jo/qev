"""Optional early exit: a distilled option readout on a truncated text stack."""

from pathlib import Path

import torch
from torch import nn


class ExitHead(nn.Module):
    """Option logits from the raw layer-L residual at the answer and condition positions.

    Submodule names follow the distillation script, so its state dicts load unchanged.
    """

    def __init__(self, hidden_size, options=16, width=1024):
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("exit head hidden size must be a positive integer")
        self.n = nn.LayerNorm(2 * hidden_size)
        self.f = nn.Sequential(
            nn.Linear(2 * hidden_size, width), nn.GELU(), nn.Linear(width, options)
        )

    def forward(self, states, condition_states):
        return self.f(self.n(torch.cat([states, condition_states], -1)))


def parse_early_exit(block, total_layers):
    """Validate a manifest ``early_exit`` block; returns (layer, head file, optional sha256)."""
    if not isinstance(block, dict) or not {"layer", "head"} <= set(block) <= {
        "layer",
        "head",
        "sha256",
    }:
        raise ValueError("early_exit requires layer and head, with an optional sha256")
    layer, head = block["layer"], block["head"]
    if type(layer) is not int or not 1 <= layer <= total_layers:
        raise ValueError(f"early_exit layer must be an integer from 1 to {total_layers}")
    # The head is a sibling file of manifest.json, never a path elsewhere.
    if not isinstance(head, str) or Path(head).name != head or not head.endswith(".safetensors"):
        raise ValueError("early_exit head must be a .safetensors file in the checkpoint")
    return layer, head, block.get("sha256")


def truncate_text_stack(language_model, layer):
    """Keep the first ``layer`` decoder layers and expose their raw residual stream.

    The final norm is removed because exit heads are distilled on un-normalized layer outputs
    (``hidden_states[layer]`` of the full model). Hybrid layer types are truncated alike.
    """
    language_model.layers = language_model.layers[:layer]
    config = language_model.config
    config.num_hidden_layers = layer
    if getattr(config, "layer_types", None):
        config.layer_types = config.layer_types[:layer]
    language_model.norm = nn.Identity()
