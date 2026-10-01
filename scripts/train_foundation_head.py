"""Compare five probability objectives on the same frozen representations and groups."""

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from veyra.probability import Calibration
from veyra.proper_learning import (
    distribution_losses,
    foundation_selection,
    grouped_batches,
    retention_losses,
    transport_losses,
)
from veyra.transfer_learning import TransferReadout, transfer_metrics

SHARES = {
    "text_nli": 0.20,
    "text_intent": 0.20,
    "image_waste": 0.25,
    "image_leaf": 0.10,
    "retention": 0.20,
    "uncertainty": 0.05,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")


@torch.inference_mode()
def evaluate(model, tensors, rows, indices):
    model.eval()
    logits = model(tensors["states"][indices], tensors["condition_logits"][indices])
    selected = [rows[i] for i in indices.tolist()]

    def summarize(mask):
        return transfer_metrics(
            logits[mask],
            tensors["targets"][indices][mask],
            tensors["valid"][indices][mask],
            tensors["gold"][indices][mask],
            tensors["types"][indices][mask],
        )

    result = {"overall": summarize(torch.ones(len(indices), dtype=torch.bool))}
    for field in ("domain", "type"):
        result["by_" + field] = {
            value: summarize(torch.tensor([row[field] == value for row in selected]))
            for value in sorted({row[field] for row in selected})
        }
    for view in ("recognition", "policy", "proposition", "partial_evidence"):
        mask = torch.tensor(["view:" + view in row.get("tags", []) for row in selected])
        if mask.any():
            result.setdefault("by_view", {})[view] = summarize(mask)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[83, 89, 97])
    parser.add_argument(
        "--methods", nargs="+", default=["ce", "direct", "direct_noise", "rloo", "cross"]
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("experiment output must be empty")
    if args.epochs < 1 or not args.seeds or not args.methods:
        raise ValueError("empty experiment")
    cache = json.loads((args.features / "manifest.json").read_text())
    spec = json.loads((args.features / "specification.json").read_text())
    if not cache.get("complete") or spec["calibration_or_final_encoded"] is not False:
        raise ValueError("a complete train/dev-only cache is required")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(args.features / name) != cache[key]:
            raise ValueError("cache fingerprint mismatch")
    if (
        digest(args.parent / "head.safetensors") != spec["parent_weights_sha256"]
        or digest(args.parent / "manifest.json") != spec["parent_manifest_sha256"]
    ):
        raise ValueError("cache parent mismatch")
    parent = json.loads((args.parent / "manifest.json").read_text())
    state = load_file(args.parent / "head.safetensors")
    tensors = load_file(args.features / "features.safetensors")
    rows = json.loads((args.features / "records.json").read_text())
    if any(r["split"] not in {"train", "dev"} for r in rows):
        raise ValueError("holdout found in training cache")
    train = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "train"])
    dev = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "dev"])
    if {rows[i]["group"] for i in train.tolist()} & {rows[i]["group"] for i in dev.tolist()}:
        raise ValueError("train/dev groups overlap")
    torch.set_num_threads(4)
    configuration = {
        "arguments": vars(args),
        "domain_shares": SHARES,
        "cache_specification": spec,
        "learning_rates": {"readout": 0.0001, "binding": 0.001},
        "batch_size": 128,
        "reward": "log + 0.5 spherical - 0.25 ordinal normalized RPS",
        "noise": {"sigma": 0.15, "independent_samples": 4, "ce_guidance": 0.5},
        "cross_weight": 0.1,
        "retention": "2*KL at temperature 2 on retention only",
        "selection": (
            "Eligible domain-macro accuracy, then macro NLL; seed-median method selection; "
            "first declared seed represents the selected method"
        ),
        "eligibility": (
            "At most 2pp regression per core domain and retention; "
            "uncertainty NLL at most baseline+0.02"
        ),
        "source_sha256": {
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path("src/veyra/proper_learning.py"),
                Path("src/veyra/transfer_learning.py"),
            ]
        },
        "calibration_or_final_used": False,
    }
    write(args.output / "protocol.json", configuration)
    initial_model = TransferReadout(parent, state).eval()
    with torch.inference_mode():
        reconstructed = initial_model(tensors["states"][dev], tensors["condition_logits"][dev])
        difference = float((reconstructed - tensors["parent_logits"][dev]).abs().max())
    if difference > 0.0005:
        raise ValueError(f"CPU parent reconstruction error: {difference}")
    initial = evaluate(initial_model, tensors, rows, dev)
    write(args.output / "baseline.json", {"dev": initial, "cpu_parent_logit_error": difference})
    print(json.dumps({"initial_dev": initial}), flush=True)
    counts = Counter(rows[i]["domain"] for i in train.tolist())
    weights = torch.zeros(len(rows))
    for i in train.tolist():
        weights[i] = SHARES[rows[i]["domain"]] * len(train) / counts[rows[i]["domain"]]
    retention = torch.tensor([r["domain"] == "retention" for r in rows])
    results = []
    for method in args.methods:
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = TransferReadout(parent, state)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.readout.parameters(), "lr": 0.0001, "weight_decay": 0.0},
                    {"params": model.binding_head.parameters(), "lr": 0.001, "weight_decay": 0.01},
                ]
            )
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs, eta_min=0.00001
            )
            best_key = foundation_selection(initial, initial)
            best_state, best_metrics, best_epoch = copy.deepcopy(model.state_dict()), initial, 0
            history, steps = [], 0
            started = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                model.train()
                total, seen = 0.0, 0
                rng = torch.Generator().manual_seed(seed * 1000 + epoch)
                for batch in grouped_batches(train, rows, 128, rng):
                    logits = model(tensors["states"][batch], tensors["condition_logits"][batch])
                    valid = tensors["valid"][batch]
                    losses = distribution_losses(
                        logits, tensors["targets"][batch], valid, tensors["types"][batch], method
                    )
                    losses = (
                        losses
                        + retention_losses(logits, tensors["parent_logits"][batch], valid)
                        * retention[batch]
                    )
                    if method == "cross":
                        losses = losses + 0.1 * transport_losses(logits, valid, batch, rows)
                    loss = (losses * weights[batch]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite objective")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    steps += 1
                    total += float(loss.detach()) * len(batch)
                    seen += len(batch)
                schedule.step()
                metrics = evaluate(model, tensors, rows, dev)
                key = foundation_selection(metrics, initial)
                history.append(
                    {"epoch": epoch, "train_loss": total / seen, "selection": key, "dev": metrics}
                )
                if key > best_key:
                    best_key, best_epoch, best_metrics = key, epoch, metrics
                    best_state = copy.deepcopy(model.state_dict())
                write(
                    args.output / "progress.json",
                    {
                        "method": method,
                        "seed": seed,
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "selected_epoch": best_epoch,
                        "selection": best_key,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
            folder = args.output / f"{method}-{seed}"
            checkpoint = folder / "checkpoint"
            checkpoint.mkdir(parents=True)
            weights_state = dict(state)
            weights_state.update({k: v.contiguous() for k, v in best_state.items()})
            save_file(weights_state, checkpoint / "head.safetensors")
            manifest = copy.deepcopy(parent)
            manifest["weights_sha256"] = digest(checkpoint / "head.safetensors")
            manifest["calibration"] = Calibration().to_dict()
            manifest["training"] = {
                "completed": True,
                "intermediate": True,
                "optimizer_steps": steps,
                "method": method,
                "seed": seed,
                "epochs": args.epochs,
                "selected_epoch": best_epoch,
                "initial_dev": initial,
                "selected_dev": best_metrics,
                "selection": best_key,
                "parent_training": parent["training"],
                "initialization_sha256": parent["weights_sha256"],
                "dataset_sha256": spec["records_sha256"],
                "protocol_sha256": digest(args.output / "protocol.json"),
                "backbone_and_condition_readout_frozen": True,
                "elapsed_seconds": time.perf_counter() - started,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "final_evaluated_at_selection": False,
                "smoke": False,
            }
            write(checkpoint / "manifest.json", manifest)
            write(folder / "history.json", history)
            result = {
                "method": method,
                "seed": seed,
                "selected_epoch": best_epoch,
                "selection": best_key,
                "dev": best_metrics,
                "checkpoint": str(checkpoint),
                "elapsed_seconds": time.perf_counter() - started,
            }
            results.append(result)
            write(args.output / "results.json", results)
            print(json.dumps(result), flush=True)
    ranking = []
    for method in args.methods:
        group = [r for r in results if r["method"] == method]
        ordered = sorted(group, key=lambda r: tuple(r["selection"]))
        ranking.append(
            {"method": method, "median_selection": ordered[len(ordered) // 2]["selection"]}
        )
    method = max(ranking, key=lambda r: tuple(r["median_selection"]))["method"]
    representative = next(
        r for r in results if r["method"] == method and r["seed"] == args.seeds[0]
    )
    write(
        args.output / "selection.json",
        {
            "ranking": ranking,
            "selected": representative,
            "no_final_or_calibration_used": True,
            "seed_policy": "First predeclared seed, not best-performing seed",
        },
    )
    print(json.dumps({"selected": representative["checkpoint"], "ranking": ranking}), flush=True)


if __name__ == "__main__":
    main()
