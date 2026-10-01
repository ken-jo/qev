"""Temperature fitting and explicit empirical-risk abstention on held-out groups."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from veyra.checkpoint import load_head, save_checkpoint
from veyra.constants import QUESTION_TYPES
from veyra.features import FeatureSample, load_features
from veyra.probability import Calibration
from veyra.training import predict_features


def fit_temperature(samples: list[FeatureSample], logits: list[torch.Tensor]) -> float:
    if not samples or len(samples) != len(logits):
        raise ValueError("temperature fitting requires aligned calibration samples")
    count = max(len(value) for value in logits)
    padded = torch.full((len(logits), count), -torch.inf, dtype=torch.float64)
    targets = torch.zeros_like(padded)
    valid = torch.zeros_like(padded, dtype=torch.bool)
    for i, (sample, value) in enumerate(zip(samples, logits, strict=True)):
        padded[i, : len(value)] = value.double()
        targets[i, : len(value)] = sample.targets.double()
        valid[i, : len(value)] = True
    grid = torch.logspace(-1, 1, 161, dtype=torch.float64)
    log_probs = torch.log_softmax(padded.unsqueeze(0) / grid[:, None, None], -1)
    log_probs = log_probs.masked_fill(~valid.unsqueeze(0), 0)
    losses = -(targets.unsqueeze(0) * log_probs).sum(-1).mean(-1)
    return float(grid[int(losses.argmin())])


def fit_abstention(
    samples: list[FeatureSample],
    logits: list[torch.Tensor],
    temperature: float,
    max_error: float = 0.15,
    minimum_accepted: int = 20,
) -> dict:
    if not 0 <= max_error <= 1 or minimum_accepted < 1:
        raise ValueError("invalid abstention policy limits")
    observations = []
    for sample, value in zip(samples, logits, strict=True):
        if not bool((sample.targets == 1).any()):
            continue
        probs = torch.softmax(value.double() / temperature, -1)
        winner = int(probs.argmax())
        observations.append((float(probs[winner]), 1.0 - float(sample.targets[winner])))
    best = None
    for threshold in sorted({confidence for confidence, _ in observations}):
        accepted = [error for confidence, error in observations if confidence >= threshold]
        if len(accepted) >= minimum_accepted and sum(accepted) / len(accepted) <= max_error:
            best = {
                "threshold": threshold,
                "accepted": len(accepted),
                "observed_error": sum(accepted) / len(accepted),
                "always_abstain": False,
            }
            break
    return best or {
        "threshold": 1.0,
        "accepted": 0,
        "observed_error": None,
        "always_abstain": True,
    }


def calibrate_checkpoint(
    records: Path,
    features: Path,
    checkpoint: Path,
    output: Path,
    device: str = "cuda",
    max_error: float = 0.15,
) -> dict:
    head, encoder_config, _, manifest = load_head(checkpoint)
    all_samples, feature_manifest = load_features(records, features)
    if manifest["training"]["feature_fingerprint"] != feature_manifest["fingerprint"]:
        raise ValueError("calibration data does not match the checkpoint feature corpus")
    samples = [sample for sample in all_samples if sample.split == "calibration"]
    if not samples:
        raise ValueError("a separate calibration split is required")
    logits = predict_features(head.to(device), samples, device=device)
    temperature_indices, policy_indices = [], []
    for i, sample in enumerate(samples):
        bucket = int(hashlib.sha256(sample.group_id.encode()).hexdigest()[:8], 16) % 2
        (temperature_indices if bucket == 0 else policy_indices).append(i)
    temperatures, thresholds, fitted, policy_fitted, always, policies = [], [], [], [], [], {}
    for type_id, name in enumerate(QUESTION_TYPES):
        fitting = [i for i in temperature_indices if samples[i].type_id == type_id]
        policy = [i for i in policy_indices if samples[i].type_id == type_id]
        if not fitting or not policy:
            raise ValueError(f"insufficient independent calibration groups for {name}")
        temperature = fit_temperature([samples[i] for i in fitting], [logits[i] for i in fitting])
        result = fit_abstention(
            [samples[i] for i in policy], [logits[i] for i in policy], temperature, max_error
        )
        temperatures.append(temperature)
        thresholds.append(result["threshold"])
        fitted.append(name)
        policy_fitted.append(name)
        if result["always_abstain"]:
            always.append(name)
        policies[name] = {
            **result,
            "policy_questions": len(policy),
            "temperature_questions": len(fitting),
        }
    calibration = Calibration(
        tuple(temperatures), tuple(thresholds), tuple(fitted), tuple(policy_fitted), tuple(always)
    )
    training = {
        **manifest["training"],
        "calibration_run": {
            "split": "calibration",
            "group_disjoint_temperature_and_policy": True,
            "policy_method": (
                "maximum coverage under empirical error limit on separate calibration groups"
            ),
            "target_empirical_error": max_error,
            "minimum_accepted": 20,
            "policies": policies,
            "limitation": (
                "Empirical calibration-set selection does not guarantee unseen-domain risk."
            ),
        },
    }
    return save_checkpoint(output, head, encoder_config, calibration, training)
