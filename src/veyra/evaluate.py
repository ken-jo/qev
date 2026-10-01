"""Probability-quality and coverage reports, with an explicit evaluation split."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from veyra.checkpoint import load_head
from veyra.constants import QUESTION_TYPES
from veyra.features import load_features
from veyra.training import predict_features, summarize_predictions


def evaluate_checkpoint(
    records: Path,
    features: Path,
    checkpoint: Path,
    output: Path,
    split: str = "dev",
    device: str = "cuda",
) -> dict:
    if split not in {"dev", "calibration", "test"}:
        raise ValueError("evaluation split must be dev, calibration, or test")
    head, _, calibration, manifest = load_head(checkpoint)
    all_samples, feature_manifest = load_features(records, features)
    if manifest["training"]["feature_fingerprint"] != feature_manifest["fingerprint"]:
        raise ValueError("evaluation features do not match the trained corpus")
    samples = [sample for sample in all_samples if sample.split == split]
    logits = predict_features(head.to(device), samples, device=device)

    def summary(indices):
        subset, predictions = [samples[i] for i in indices], [logits[i] for i in indices]
        metrics = summarize_predictions(subset, predictions, calibration.temperatures)
        accepted, accepted_correctness = [], []
        for sample, value in zip(subset, predictions, strict=True):
            probs = torch.softmax(value.double() / calibration.temperatures[sample.type_id], -1)
            confidence, winner = float(probs.max()), int(probs.argmax())
            accept = (
                QUESTION_TYPES[sample.type_id] not in calibration.always_abstain_types
                and confidence >= calibration.abstain_thresholds[sample.type_id]
            )
            accepted.append(accept)
            if accept:
                accepted_correctness.append(float(sample.targets[winner]))
        metrics.update(
            {
                "coverage": sum(accepted) / len(accepted),
                "accepted_questions": sum(accepted),
                "accepted_expected_accuracy": (
                    sum(accepted_correctness) / len(accepted_correctness)
                    if accepted_correctness
                    else None
                ),
            }
        )
        return metrics

    report = {
        "split": split,
        "feature_fingerprint": feature_manifest["fingerprint"],
        "checkpoint_weights_sha256": manifest["weights_sha256"],
        "calibration": calibration.to_dict(),
        "overall": summary(list(range(len(samples)))),
        "raw_overall": summarize_predictions(samples, logits),
        "uniform_probability_baseline": summarize_predictions(
            samples, [torch.zeros_like(x) for x in logits]
        ),
        "random_choice_expected_accuracy": sum(1 / len(x) for x in logits) / len(logits),
        "by_type": {
            name: summary([i for i, sample in enumerate(samples) if sample.type_id == type_id])
            for type_id, name in enumerate(QUESTION_TYPES)
            if any(sample.type_id == type_id for sample in samples)
        },
        "by_family": {
            name: summary([i for i, sample in enumerate(samples) if sample.family == name])
            for name in sorted({sample.family for sample in samples})
        },
        "by_language": {
            name: summary([i for i, sample in enumerate(samples) if sample.language == name])
            for name in sorted({sample.language for sample in samples})
        },
        "by_tag": {
            name: summary([i for i, sample in enumerate(samples) if name in sample.tags])
            for name in sorted({tag for sample in samples for tag in sample.tags})
        },
        "by_cardinality": {
            str(count): summary(
                [i for i, sample in enumerate(samples) if len(sample.targets) == count]
            )
            for count in sorted({len(sample.targets) for sample in samples})
        },
        "notes": [
            "Soft targets contribute target mass at the selected alternative to expected accuracy.",
            "Hard-label accuracy excludes soft target questions.",
            "Uniform baseline argmax breaks ties at the first candidate; "
            "random choice is reported separately.",
            "Coverage is empirical on this split and is not a guarantee for unseen inputs.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"overall": report["overall"], "by_family": report["by_family"]}, indent=2),
        flush=True,
    )
    return report
