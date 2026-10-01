"""Adapt the shared late backbone on train groups, with predeclared dev checkpoints."""

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_foundation_head import SHARES, evaluate, write

from veyra.data import read_records
from veyra.option_model import OptionModel, option_prompt
from veyra.proper_learning import distribution_losses, foundation_selection, retention_losses


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(model, record, root):
    output, _ = model(record.request, root)
    if len(output) != 1:
        raise ValueError("foundation records must contain one question")
    name = next(iter(output))
    _, positions = option_prompt(record.request, record.request.questions[name])
    values = torch.zeros(16, device=model.encoder.device)
    return values.index_copy(0, torch.tensor(positions, device=values.device), output[name])


@torch.inference_mode()
def assess(model, records, root, rows, tensors, indices):
    model.eval()
    values = torch.stack(
        [canonical(model, records[rows[i]["id"]], root).cpu() for i in indices.tolist()]
    )

    class Predictions:
        def eval(self):
            return self

        def __call__(self, states, conditions):
            return values

    return evaluate(Predictions(), tensors, rows, indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--head-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.00003)
    parser.add_argument("--seed", type=int, default=107)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("training output must be empty")
    if args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("invalid training budget")
    selected = json.loads((args.head_experiment / "selection.json").read_text())["selected"]
    parent = Path(selected["checkpoint"])
    method = selected["method"]
    baseline = json.loads((args.head_experiment / "baseline.json").read_text())["dev"]
    tensors = load_file(args.features / "features.safetensors")
    rows = json.loads((args.features / "records.json").read_text())
    spec = json.loads((args.features / "specification.json").read_text())
    if digest(args.records) != spec["records_sha256"]:
        raise ValueError("dataset changed after feature extraction")
    records = {r.id: r for r in read_records(args.records) if r.split in {"train", "dev"}}
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    dev = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "dev"])
    groups = {}
    for i in train:
        groups.setdefault(rows[i]["group"], []).append(i)
    counts = Counter(rows[i]["domain"] for i in train)
    weights = {d: SHARES[d] * len(train) / n for d, n in counts.items()}
    protocol = {
        "arguments": vars(args),
        "parent": str(parent),
        "method": method,
        "parent_weights_sha256": digest(parent / "head.safetensors"),
        "dataset_sha256": digest(args.records),
        "domain_shares": SHARES,
        "accumulation": 8,
        "head_lr_multiplier": 0.1,
        "selection": "Original-release domain guardrails, then domain macro accuracy and NLL",
        "candidates": (
            "Selected head, each half-epoch and full epoch; deployment merge evaluated later"
        ),
        "condition_readout_frozen": True,
        "vision_backbone_frozen": True,
        "calibration_or_final_used": False,
        "source_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("src/veyra/proper_learning.py"),
                Path("src/veyra/option_model.py"),
            )
        },
    }
    write(args.output / "protocol.json", protocol)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = OptionModel.load(parent, local_files_only=True, merge=False)
    model.condition_readout.requires_grad_(False)
    adapters = [p for name, p in model.named_parameters() if ".lora_" in name]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapters, "lr": args.learning_rate},
            {
                "params": list(model.readout.parameters()) + list(model.binding_head.parameters()),
                "lr": args.learning_rate * 0.1,
            },
        ],
        weight_decay=0.01,
    )
    total_steps = math.ceil(len(train) / 8) * args.epochs

    def rate(step):
        if step < 30:
            return (step + 1) / 30
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1, (step - 30) / max(1, total_steps - 30))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    history, steps = [], 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    for epoch in range(1, args.epochs + 1):
        units = list(groups.values())
        random.Random(args.seed * 1000 + epoch).shuffle(units)
        # Recognition precedes its derived views; all views of a group stay together.
        order = [
            i for unit in units for i in sorted(unit, key=lambda j: "transport_parent" in rows[j])
        ]
        checkpoints = {len(order) // 2, len(order)}
        optimizer.zero_grad(set_to_none=True)
        model.train()
        model.encoder.model.eval()
        probabilities, group, total = {}, None, 0.0
        for offset, i in enumerate(order, start=1):
            if rows[i]["group"] != group:
                probabilities, group = {}, rows[i]["group"]
            logits = canonical(model, records[rows[i]["id"]], args.records.parent).unsqueeze(0)
            valid = tensors["valid"][i : i + 1].to(logits.device)
            target = tensors["targets"][i : i + 1].to(logits.device)
            types = tensors["types"][i : i + 1].to(logits.device)
            loss = distribution_losses(logits, target, valid, types, method).mean()
            if rows[i]["domain"] == "retention":
                loss = (
                    loss
                    + retention_losses(
                        logits, tensors["parent_logits"][i : i + 1].to(logits.device), valid
                    ).mean()
                )
            probabilities[i] = logits.detach().masked_fill(~valid, -1e9).softmax(-1)[0]
            if method == "cross" and "transport_parent" in rows[i]:
                source = probabilities[rows[i]["transport_parent"]]
                pushed = torch.zeros(16, device=logits.device)
                for origin, destination in enumerate(rows[i]["transport_map"]):
                    if destination >= 0:
                        pushed[destination] += source[origin]
                loss = (
                    loss
                    + 0.1
                    * (
                        pushed
                        * (
                            pushed.clamp_min(1e-12).log()
                            - logits[0].masked_fill(~valid[0], -1e9).log_softmax(-1)
                        )
                    ).sum()
                )
            loss = loss * weights[rows[i]["domain"]]
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training objective")
            # Flush at the midpoint before evaluation as well as at regular windows.
            boundary = min(((offset - 1) // 8 + 1) * 8, len(order))
            midpoint = len(order) // 2
            if offset <= midpoint < boundary:
                boundary = midpoint
            start = ((offset - 1) // 8) * 8
            if start < midpoint < offset:
                start = midpoint
            (loss / (boundary - start)).backward()
            total += float(loss.detach())
            if offset == boundary:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            if offset % 200 == 0:
                progress = {
                    "epoch": epoch,
                    "processed": offset,
                    "total": len(order),
                    "steps": steps,
                    "mean_loss": total / offset,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                write(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if offset in checkpoints:
                metrics = assess(model, records, args.records.parent, rows, tensors, dev)
                fraction = 0.5 if offset < len(order) else 1.0
                path = args.output / f"epoch-{epoch - 1 + fraction:g}"
                metadata = {
                    "completed": True,
                    "intermediate": True,
                    "optimizer_steps": steps,
                    "method": method,
                    "backbone_updated": True,
                    "epochs": epoch - 1 + fraction,
                    "seed": args.seed,
                    "parent_training": parent_manifest["training"],
                    "dataset_sha256": digest(args.records),
                    "protocol_sha256": digest(args.output / "protocol.json"),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "selected_dev": metrics,
                    "selection": foundation_selection(metrics, baseline),
                    "final_evaluated_at_selection": False,
                    "smoke": False,
                }
                model.save(path, metadata)
                history.append({"checkpoint": str(path), **metadata})
                write(args.output / "history.json", history)
                print(json.dumps({"checkpoint": str(path), "dev": metrics}), flush=True)
                model.train()
                model.encoder.model.eval()
    write(
        args.output / "complete.json",
        {
            "completed": True,
            "candidates": [str(parent)] + [h["checkpoint"] for h in history],
            "requires_merged_dev_selection": True,
        },
    )


if __name__ == "__main__":
    main()
