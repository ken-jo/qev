"""Fit a small residual readout on frozen deployment features, selecting on development only."""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from veyra.binding_head import ConditionedReadout


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def evaluate(head, tensors, rows, indices):
    head.eval()
    delta = head(tensors["states"][indices], tensors["condition_logits"][indices])
    logits = (tensors["base_logits"][indices] + delta).masked_fill(
        ~tensors["valid"][indices], -torch.inf
    )
    target = tensors["targets"][indices]
    log_probs = logits.log_softmax(-1).masked_fill(~tensors["valid"][indices], 0)
    correct = target.gather(1, logits.argmax(-1, keepdim=True)).squeeze(1)
    nll = -(target * log_probs).sum(-1)
    selected = [rows[index] for index in indices.tolist()]
    result = {
        "questions": len(selected),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
    }
    for kind in ("text", "image"):
        mask = torch.tensor([row["family"].startswith(kind + "_") for row in selected])
        result[kind + "_accuracy"] = float(correct[mask].mean())
    for field in ("family", "type"):
        result["by_" + field] = {
            kind: float(correct[torch.tensor([row[field] == kind for row in selected])].mean())
            for kind in sorted({row[field] for row in selected})
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["condition", "uniform"], required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--residual-penalty", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("training output must be empty")
    if min(args.epochs, args.batch_size, args.learning_rate) <= 0 or args.residual_penalty < 0:
        raise ValueError("invalid training settings")
    cache = json.loads((args.features / "manifest.json").read_text())
    specification = json.loads((args.features / "specification.json").read_text())
    if not cache.get("complete") or specification.get("calibration_or_final_encoded") is not False:
        raise ValueError("require a completed train/development-only cache")
    for name, field in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if file_hash(args.features / name) != cache[field]:
            raise ValueError("feature cache checksum mismatch")
    parent = json.loads((args.parent / "manifest.json").read_text())
    if (
        file_hash(args.parent / "manifest.json") != specification["parent_manifest_sha256"]
        or file_hash(args.parent / "head.safetensors") != specification["parent_weights_sha256"]
        or parent.get("binding_head", {}).get("rank", 0)
    ):
        raise ValueError("feature cache and parent checkpoint disagree")
    rows = json.loads((args.features / "records.json").read_text())
    if any(row["split"] not in {"train", "dev"} for row in rows):
        raise ValueError("residual fitting cannot consume calibration or final data")
    train_groups = {row["group"] for row in rows if row["split"] == "train"}
    dev_groups = {row["group"] for row in rows if row["split"] == "dev"}
    if train_groups & dev_groups:
        raise ValueError("training and development groups overlap")
    tensors = load_file(args.features / "features.safetensors")
    train = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "train"])
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    if not len(train) or not len(dev) or len(rows) != tensors["states"].shape[0]:
        raise ValueError("cache rows are missing or inconsistent")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    head = ConditionedReadout(args.rank, args.mode)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = math.ceil(len(train) / args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps, eta_min=args.learning_rate * 0.1
    )
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(path): file_hash(path)
        for path in (
            Path(__file__),
            Path("src/veyra/binding_head.py"),
            Path("src/veyra/option_model.py"),
        )
    }
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "arguments": {**vars(args), "selection": "min-modality-accuracy"},
                "source_sha256": source_hashes,
            },
            default=str,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    initial = evaluate(head, tensors, rows, dev)
    print(json.dumps({"initial_dev": initial}), flush=True)
    best_value = (min(initial["text_accuracy"], initial["image_accuracy"]), -initial["nll"])
    best = copy.deepcopy(head.state_dict())
    selected_epoch, steps, history = 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = train[torch.randperm(len(train))]
        head.train()
        total_loss = 0.0
        for batch in order.split(args.batch_size):
            delta = head(tensors["states"][batch], tensors["condition_logits"][batch])
            valid = tensors["valid"][batch]
            logits = (tensors["base_logits"][batch] + delta).masked_fill(~valid, -torch.inf)
            log_probs = logits.log_softmax(-1).masked_fill(~valid, 0)
            loss = -(tensors["targets"][batch] * log_probs).sum(-1).mean()
            loss = loss + args.residual_penalty * delta.square()[valid].mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite residual loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * len(batch)
            steps += 1
        metrics = evaluate(head, tensors, rows, dev)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), "dev": metrics})
        value = (min(metrics["text_accuracy"], metrics["image_accuracy"]), -metrics["nll"])
        if value > best_value:
            best_value, best, selected_epoch = value, copy.deepcopy(head.state_dict()), epoch
        print(json.dumps(history[-1]), flush=True)
    state = load_file(args.parent / "head.safetensors")
    state.update({"binding_head." + name: value.contiguous() for name, value in best.items()})
    checkpoint = args.output / "checkpoint"
    checkpoint.mkdir()
    save_file(state, checkpoint / "head.safetensors")
    manifest = copy.deepcopy(parent)
    manifest["binding_head"] = {"rank": args.rank, "mode": args.mode}
    manifest["weights_sha256"] = file_hash(checkpoint / "head.safetensors")
    manifest["training"] = {
        "completed": True,
        "intermediate": True,
        "optimizer_steps": steps,
        "epochs": args.epochs,
        "selected_epoch": selected_epoch,
        "selection_split": "dev",
        "selection_metric": "min-modality-accuracy",
        "initial_dev": initial,
        "history": history,
        "initialization_sha256": parent["weights_sha256"],
        "parent_training": parent["training"],
        "backbone_and_condition_readout_frozen": True,
        "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "cache_manifest_sha256": file_hash(args.features / "manifest.json"),
        "cache_specification": specification,
        "source_sha256": source_hashes,
        "smoke": specification.get("smoke", False),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (checkpoint / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"saved": str(checkpoint), "selected_epoch": selected_epoch, "selection": best_value}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
