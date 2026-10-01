"""Supervised frozen-backbone training with development-only checkpoint selection."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import torch
from torch import Tensor

from veyra.backbone import EncoderConfig
from veyra.checkpoint import save_checkpoint
from veyra.features import FeatureSample, collate, load_features
from veyra.head import DecisionHead, HeadConfig
from veyra.probability import Calibration


def distribution_loss(logits: Tensor, targets: Tensor, valid: Tensor, types: Tensor) -> Tensor:
    log_probs = torch.log_softmax(logits, -1).masked_fill(~valid, 0)
    ce = -(targets * log_probs).sum(-1)
    probabilities = torch.softmax(logits, -1)
    ranked = ((probabilities.cumsum(-1) - targets.cumsum(-1)) ** 2).sum(-1)
    ranked = ranked / (valid.sum(-1) - 1)
    return (ce + 0.25 * ranked * (types == 1)).mean()


def head_forward(head: DecisionHead, batch: dict) -> Tensor:
    return head(
        batch["features"],
        batch["valid"],
        batch["question_types"],
        batch["levels"],
        batch["context"],
    )


@torch.inference_mode()
def predict_features(
    head: DecisionHead, samples: list[FeatureSample], batch_size: int = 64, device: str = "cpu"
) -> list[Tensor]:
    head.eval()
    results = []
    for start in range(0, len(samples), batch_size):
        subset = samples[start : start + batch_size]
        batch = collate(subset, device)
        logits = head_forward(head, batch).cpu()
        results.extend(
            logits[i, : sample.targets.numel()].clone() for i, sample in enumerate(subset)
        )
    return results


def summarize_predictions(
    samples: list[FeatureSample], logits: list[Tensor], temperatures=(1, 1, 1)
) -> dict:
    if not samples or len(samples) != len(logits):
        raise ValueError("samples and logits must be nonempty and aligned")
    nll, brier, correctness, confidence, hard_correct, score_mae = [], [], [], [], [], []
    for sample, prediction in zip(samples, logits, strict=True):
        probs = torch.softmax(prediction.double() / temperatures[sample.type_id], -1)
        target = sample.targets.double()
        nll.append(float(-(target * probs.clamp_min(1e-12).log()).sum()))
        brier.append(float(((probs - target) ** 2).sum()))
        winner = int(probs.argmax())
        correctness.append(float(target[winner]))
        confidence.append(float(probs[winner]))
        if bool((target == 1).any()):
            hard_correct.append(float(target[winner]))
        if sample.type_id == 1:
            levels = torch.arange(len(probs), dtype=torch.float64)
            score_mae.append(float(abs((probs * levels).sum() - (target * levels).sum())))
    ece = 0.0
    for index in range(10):
        members = [i for i, value in enumerate(confidence) if min(int(value * 10), 9) == index]
        if members:
            gap = abs(sum(confidence[i] - correctness[i] for i in members) / len(members))
            ece += len(members) / len(samples) * gap
    return {
        "questions": len(samples),
        "nll": sum(nll) / len(nll),
        "brier": sum(brier) / len(brier),
        "expected_accuracy": sum(correctness) / len(correctness),
        "hard_label_accuracy": sum(hard_correct) / len(hard_correct) if hard_correct else None,
        "hard_label_questions": len(hard_correct),
        "ece_10_bins": ece,
        "score_mae": sum(score_mae) / len(score_mae) if score_mae else None,
    }


def train_head(
    records: Path,
    features: Path,
    output: Path,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 0.0003,
    device: str = "cuda",
    seed: int = 19,
) -> dict:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("training settings must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("training checkpoint output must be empty")
    torch.manual_seed(seed)
    start_time = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    randomizer = random.Random(seed)
    all_samples, feature_manifest = load_features(records, features)
    train = [sample for sample in all_samples if sample.split == "train"]
    dev = [sample for sample in all_samples if sample.split == "dev"]
    if not train or not dev:
        raise ValueError("training and development splits are required")
    head = DecisionHead(HeadConfig(input_size=train[0].features.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    initial = summarize_predictions(dev, predict_features(head, dev, batch_size, device))
    best_nll = float("inf")
    best_state, best_epoch = None, 0
    history = []
    steps = 0
    for epoch in range(1, epochs + 1):
        order = list(range(len(train)))
        randomizer.shuffle(order)
        head.train()
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            subset = [train[index] for index in order[start : start + batch_size]]
            batch = collate(subset, device)
            optimizer.zero_grad(set_to_none=True)
            logits = head_forward(head, batch)
            loss = distribution_loss(
                logits, batch["targets"], batch["valid"], batch["question_types"]
            )
            if not torch.isfinite(loss):
                raise ValueError("training produced nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(subset)
            steps += 1
        metrics = summarize_predictions(dev, predict_features(head, dev, batch_size, device))
        history.append({"epoch": epoch, "training_loss": total_loss / len(train), "dev": metrics})
        print(json.dumps(history[-1]), flush=True)
        if metrics["nll"] < best_nll:
            best_nll, best_epoch = metrics["nll"], epoch
            best_state = copy.deepcopy(
                {name: tensor.cpu() for name, tensor in head.state_dict().items()}
            )
    head.load_state_dict(best_state)
    training = {
        "completed": True,
        "optimizer_steps": steps,
        "epochs": epochs,
        "selected_epoch": best_epoch,
        "selection_split": "dev",
        "selection_metric": "nll",
        "selected_dev_nll": best_nll,
        "train_questions": len(train),
        "dev_questions": len(dev),
        "seed": seed,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "batch_size": batch_size,
        "loss": "cross_entropy + 0.25 * ranked_probability_score for score questions",
        "elapsed_seconds": time.perf_counter() - start_time,
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if device.startswith("cuda")
        else None,
        "memory_scope": "cached features and head training; backbone runs in extraction phase",
        "feature_fingerprint": feature_manifest["fingerprint"],
        "dataset_sha256": feature_manifest["specification"]["dataset_sha256"],
        "initial_dev": initial,
        "history": history,
    }
    manifest = save_checkpoint(
        output,
        head,
        EncoderConfig(**feature_manifest["specification"]["encoder"]),
        Calibration(),
        training,
    )
    print(f"Saved trained head at {output}; selected epoch {best_epoch}", flush=True)
    return manifest
