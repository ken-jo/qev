"""Exact weighted mean of LoRA updates, represented by concatenated factors."""

from __future__ import annotations

import math

import torch


def average_states(states, scales, weights):
    if len(states) < 2 or len(states) != len(scales) or len(states) != len(weights):
        raise ValueError("provide aligned states, scales, and weights for at least two models")
    if any(not math.isfinite(value) or value <= 0 for value in (*scales, *weights)):
        raise ValueError("scales and weights must be finite and positive")
    if any(set(state) != set(states[0]) for state in states):
        raise ValueError("all models must have identical adapted modules")
    total_weight = sum(weights)
    if not math.isfinite(total_weight):
        raise ValueError("weight sum must be finite")
    head_shape = states[0]["readout.weight"].shape
    if len(head_shape) != 2 or any(state["readout.weight"].shape != head_shape for state in states):
        raise ValueError("readout dimensions must match")
    weights = [value / total_weight for value in weights]
    result = {
        "readout.weight": sum(
            weight * state["readout.weight"].float()
            for state, weight in zip(states, weights, strict=True)
        )
    }
    modules = [name.removesuffix(".lora_a") for name in states[0] if name.endswith(".lora_a")]
    expected = {"readout.weight"} | {
        name + suffix for name in modules for suffix in (".lora_a", ".lora_b")
    }
    if not modules or set(states[0]) != expected:
        raise ValueError("expected a readout and paired LoRA factors")
    for name in modules:
        first, second = [], []
        in_features, out_features = None, None
        for state, scale, weight in zip(states, scales, weights, strict=True):
            a, b = state[name + ".lora_a"].float(), state[name + ".lora_b"].float()
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError("invalid LoRA factor shapes")
            if in_features is not None and (a.shape[1], b.shape[0]) != (in_features, out_features):
                raise ValueError("adapter base dimensions differ")
            in_features, out_features = a.shape[1], b.shape[0]
            first.append(a)
            second.append(b * (weight * scale))
        # The new adapter has scale 1. Matrix multiplication contains only matching
        # blocks: B_cat @ A_cat = sum_i weight_i * scale_i * B_i @ A_i.
        result[name + ".lora_a"] = torch.cat(first, dim=0).contiguous()
        result[name + ".lora_b"] = torch.cat(second, dim=1).contiguous()
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite averaged parameters")
    return result
