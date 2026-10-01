"""Shared-backbone adaptation with frozen-baseline replay and development-only checkpoints."""

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
from train_foundation_head import write

from veyra.data import read_records
from veyra.option_model import OptionModel
from veyra.proper_learning import distribution_losses, retention_losses
from veyra.workflow_learning import SHARES, annotate, selection_key, summarize_logits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward(model, record, root):
    states, mappings, _, conditions = model.encode(record.request, root, capture_condition=True)
    if states.shape[0] != 1 or len(mappings) != 1:
        raise ValueError("require one independent decision")
    logits = model.readout(states) + model.binding_head(states, model.condition_readout(conditions))
    return logits, states


@torch.inference_mode()
def assess(model, records, root, rows, tensors, indices, temperatures):
    model.eval()
    logits = torch.stack(
        [forward(model, records[rows[i]["id"]], root)[0][0].cpu() for i in indices.tolist()]
    )
    return summarize_logits(logits, tensors, rows, indices, temperatures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/workflow-v12/records.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("data/features-workflow-v12"))
    parser.add_argument("--head-experiment", type=Path, default=Path("runs/workflow-v12-head"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("backbone experiment output is immutable")
    configuration = Path("configs/workflow-release-v12.json")
    config = json.loads(configuration.read_text(encoding="utf-8"))
    if (
        args.learning_rate not in config["backbone_learning_rates"]
        or config["backbone_epochs"] != 1
    ):
        raise ValueError("run outside the frozen experiment plan")
    selected = json.loads((args.head_experiment / "selection.json").read_text(encoding="utf-8"))[
        "selected"
    ]
    parent = Path(selected["checkpoint"])
    parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    temperatures = parent_manifest["calibration"]["temperatures"]
    baseline = json.loads((args.head_experiment / "baseline.json").read_text(encoding="utf-8"))[
        "dev"
    ]
    spec = json.loads((args.features / "specification.json").read_text(encoding="utf-8"))
    if digest(args.records) != spec["records_sha256"]:
        raise ValueError("data changed after feature extraction")
    tensors = load_file(args.features / "features.safetensors")
    rows = json.loads((args.features / "records.json").read_text(encoding="utf-8"))
    records = {r.id: r for r in read_records(args.records) if r.split in {"train", "dev"}}
    annotate(rows, records.values())
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    dev = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "dev"])
    groups = {}
    for i in train:
        groups.setdefault(rows[i]["group"], []).append(i)
    units = list(groups.values())
    random.Random(config["seed"]).shuffle(units)
    order = [i for unit in units for i in unit]
    segments = [order[: len(order) // 2], order[len(order) // 2 :]]
    counts = Counter(rows[i]["domain"] for i in train)
    weights = {d: SHARES[d] * len(train) / n for d, n in counts.items()}
    protocol = {
        "arguments": vars(args),
        "release_protocol_sha256": digest(configuration),
        "parent": str(parent),
        "parent_weights_sha256": digest(parent / "head.safetensors"),
        "dataset_sha256": digest(args.records),
        "method": selected["method"],
        "seed": config["seed"],
        "domain_shares": SHARES,
        "accumulation": 8,
        "head_lr_multiplier": 0.1,
        "retention": (
            "0.25 supervised + 4*temperature-2 KL + 0.1 cosine feature anchoring on replay"
        ),
        "source_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("src/veyra/workflow_learning.py"),
                Path("src/veyra/proper_learning.py"),
                Path("src/veyra/option_model.py"),
            )
        },
        "condition_readout_frozen": True,
        "vision_weights_frozen": True,
        "final_or_calibration_used": False,
    }
    write(args.output / "protocol.json", protocol)
    torch.manual_seed(config["seed"])
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
    total_steps = sum(math.ceil(len(segment) / 8) for segment in segments)

    def rate(step):
        if step < 30:
            return (step + 1) / 30
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1, (step - 30) / max(1, total_steps - 30))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    started, processed, steps, total_loss, history = time.perf_counter(), 0, 0, 0.0, []
    torch.cuda.reset_peak_memory_stats()
    for segment_index, segment in enumerate(segments):
        model.train()
        model.encoder.model.eval()
        for start in range(0, len(segment), 8):
            batch = segment[start : start + 8]
            optimizer.zero_grad(set_to_none=True)
            for i in batch:
                logits, hidden = forward(model, records[rows[i]["id"]], args.records.parent)
                valid = tensors["valid"][i : i + 1].to(logits.device)
                types = tensors["types"][i : i + 1].to(logits.device)
                scale = torch.tensor(temperatures, device=logits.device)[types, None]
                scaled = logits / scale
                loss = distribution_losses(
                    scaled,
                    tensors["targets"][i : i + 1].to(logits.device),
                    valid,
                    types,
                    selected["method"],
                ).mean()
                if rows[i]["retention"]:
                    reference = tensors["parent_logits"][i : i + 1].to(logits.device) / scale
                    anchoring = (
                        1
                        - torch.nn.functional.cosine_similarity(
                            hidden, tensors["states"][i : i + 1].to(hidden.device)
                        ).mean()
                    )
                    loss = (
                        0.25 * loss
                        + 4 * retention_losses(scaled, reference, valid).mean()
                        + 0.1 * anchoring
                    )
                loss = loss * weights[rows[i]["domain"]]
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training objective")
                (loss / len(batch)).backward()
                total_loss += float(loss.detach())
                processed += 1
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            steps += 1
            if steps % 25 == 0:
                progress = {
                    "processed": processed,
                    "total": len(train),
                    "steps": steps,
                    "mean_loss": total_loss / processed,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                write(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        metrics = assess(model, records, args.records.parent, rows, tensors, dev, temperatures)
        path = args.output / ("half" if segment_index == 0 else "full")
        metadata = {
            "completed": True,
            "intermediate": True,
            "optimizer_steps": steps,
            "method": selected["method"],
            "seed": config["seed"],
            "epochs": 0.5 * (segment_index + 1),
            "backbone_updated": True,
            "parent_training": parent_manifest["training"],
            "dataset_sha256": digest(args.records),
            "protocol_sha256": digest(args.output / "protocol.json"),
            "selected_dev": metrics,
            "selection": selection_key(metrics, baseline),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "final_evaluated_at_selection": False,
            "smoke": False,
        }
        model.save(path, metadata)
        # OptionModel.save preserves this model's frozen parent calibration until final fitting.
        history.append({"checkpoint": str(path), **metadata})
        write(args.output / "history.json", history)
        print(
            json.dumps(
                {
                    "checkpoint": str(path),
                    "selection": metadata["selection"],
                    "domains": metrics["by_domain"],
                }
            ),
            flush=True,
        )
    write(
        args.output / "complete.json",
        {
            "completed": True,
            "candidates": [h["checkpoint"] for h in history],
            "requires_merged_dev_selection": True,
        },
    )


if __name__ == "__main__":
    main()
