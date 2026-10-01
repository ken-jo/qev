"""Compare matched objectives with retention anchoring on frozen Foundation features."""

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from train_foundation_head import write

from veyra.data import read_records
from veyra.proper_learning import distribution_losses, grouped_batches, retention_losses
from veyra.transfer_learning import TransferReadout
from veyra.workflow_learning import SHARES, annotate, evaluate, selection_key


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/features-workflow-v12"))
    parser.add_argument("--records", type=Path, default=Path("data/workflow-v12/records.jsonl"))
    parser.add_argument("--parent", type=Path, default=Path("checkpoints/veyra-foundation-v11"))
    parser.add_argument("--output", type=Path, default=Path("runs/workflow-v12-head"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("head experiment output is immutable")
    config_path = Path("configs/workflow-release-v12.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache = json.loads((args.features / "manifest.json").read_text(encoding="utf-8"))
    spec = json.loads((args.features / "specification.json").read_text(encoding="utf-8"))
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(args.features / name) != cache[key]:
            raise ValueError("cache fingerprint mismatch")
    if (
        not cache["complete"]
        or spec["calibration_or_final_encoded"]
        or digest(args.records) != spec["records_sha256"]
    ):
        raise ValueError("cache must be complete and restricted to train/development")
    if (
        digest(args.parent / "head.safetensors") != config["baseline_weights_sha256"]
        or digest(args.parent / "manifest.json") != config["baseline_manifest_sha256"]
    ):
        raise ValueError("frozen baseline changed")
    if spec["parent_weights_sha256"] != config["baseline_weights_sha256"]:
        raise ValueError("features came from a different parent")
    parent = json.loads((args.parent / "manifest.json").read_text(encoding="utf-8"))
    state = load_file(args.parent / "head.safetensors")
    tensors = load_file(args.features / "features.safetensors")
    rows = json.loads((args.features / "records.json").read_text(encoding="utf-8"))
    annotate(rows, [r for r in read_records(args.records) if r.split in {"train", "dev"}])
    if any(r["split"] not in {"train", "dev"} for r in rows):
        raise ValueError("holdout in training features")
    train = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "train"])
    dev = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "dev"])
    if {rows[i]["group"] for i in train.tolist()} & {rows[i]["group"] for i in dev.tolist()}:
        raise ValueError("observation overlap")
    temperatures = parent["calibration"]["temperatures"]
    scale = torch.tensor(temperatures)[tensors["types"], None]
    torch.set_num_threads(4)
    protocol = {
        "arguments": vars(args),
        "release_protocol_sha256": digest(config_path),
        "feature_specification": spec,
        "domain_shares": SHARES,
        "epochs": config["head_epochs"],
        "methods": config["head_methods"],
        "seeds": config["head_seeds"],
        "learning_rates": {"readout": 0.0001, "binding": 0.001},
        "batch_size": 128,
        "retention": (
            "0.25 supervised loss + 4 * existing temperature-2 KL on replay; "
            "loss and dev probabilities use frozen parent temperatures"
        ),
        "source_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("src/veyra/workflow_learning.py"),
                Path("src/veyra/proper_learning.py"),
            )
        },
        "calibration_or_final_used": False,
    }
    write(args.output / "protocol.json", protocol)
    model = TransferReadout(parent, state).eval()
    with torch.inference_mode():
        difference = float(
            (
                model(tensors["states"][dev], tensors["condition_logits"][dev])
                - tensors["parent_logits"][dev]
            )
            .abs()
            .max()
        )
    if difference > 0.0005:
        raise ValueError("CPU reconstruction error: " + str(difference))
    initial = evaluate(model, tensors, rows, dev, temperatures)
    write(args.output / "baseline.json", {"dev": initial, "reconstruction_error": difference})
    print(json.dumps({"baseline": initial["by_domain"]}), flush=True)
    counts = Counter(rows[i]["domain"] for i in train.tolist())
    weights = torch.tensor(
        [SHARES.get(r["domain"], 0) * len(train) / max(1, counts[r["domain"]]) for r in rows]
    )
    replay = torch.tensor([r["retention"] for r in rows])
    results = []
    for method in config["head_methods"]:
        for seed in config["head_seeds"]:
            torch.manual_seed(seed)
            model = TransferReadout(parent, state)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.readout.parameters(), "lr": 0.0001, "weight_decay": 0.0},
                    {"params": model.binding_head.parameters(), "lr": 0.001, "weight_decay": 0.01},
                ]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config["head_epochs"], eta_min=0.00001
            )
            best_key, best_epoch, best_metrics = selection_key(initial, initial), 0, initial
            best_state = copy.deepcopy(model.state_dict())
            started, history, steps = time.perf_counter(), [], 0
            for epoch in range(1, config["head_epochs"] + 1):
                model.train()
                total = 0.0
                for batch in grouped_batches(
                    train, rows, 128, torch.Generator().manual_seed(seed * 1000 + epoch)
                ):
                    logits = (
                        model(tensors["states"][batch], tensors["condition_logits"][batch])
                        / scale[batch]
                    )
                    losses = distribution_losses(
                        logits,
                        tensors["targets"][batch],
                        tensors["valid"][batch],
                        tensors["types"][batch],
                        method,
                    )
                    losses = losses * torch.where(replay[batch], 0.25, 1.0)
                    losses += (
                        4
                        * retention_losses(
                            logits,
                            tensors["parent_logits"][batch] / scale[batch],
                            tensors["valid"][batch],
                        )
                        * replay[batch]
                    )
                    loss = (losses * weights[batch]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite objective")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    steps += 1
                    total += float(loss.detach()) * len(batch)
                scheduler.step()
                metrics = evaluate(model, tensors, rows, dev, temperatures)
                key = selection_key(metrics, initial)
                if key > best_key:
                    best_key, best_epoch, best_metrics = key, epoch, metrics
                    best_state = copy.deepcopy(model.state_dict())
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": total / len(train),
                        "selection": key,
                        "dev": metrics,
                    }
                )
                progress = {
                    "method": method,
                    "seed": seed,
                    "epoch": epoch,
                    "selected_epoch": best_epoch,
                    "selection": best_key,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                write(args.output / "progress.json", progress)
                if epoch % 5 == 0:
                    print(json.dumps(progress), flush=True)
            folder = args.output / f"{method}-{seed}"
            checkpoint = folder / "checkpoint"
            checkpoint.mkdir(parents=True)
            weights_state = dict(state)
            weights_state.update({k: v.contiguous() for k, v in best_state.items()})
            save_file(weights_state, checkpoint / "head.safetensors")
            manifest = copy.deepcopy(parent)
            manifest["weights_sha256"] = digest(checkpoint / "head.safetensors")
            manifest["training"] = {
                "completed": True,
                "intermediate": True,
                "optimizer_steps": steps,
                "method": method,
                "seed": seed,
                "selected_epoch": best_epoch,
                "selected_dev": best_metrics,
                "selection": best_key,
                "parent_training": parent["training"],
                "dataset_sha256": spec["records_sha256"],
                "protocol_sha256": digest(args.output / "protocol.json"),
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
    ranking = []
    for method in config["head_methods"]:
        group = sorted(
            [r for r in results if r["method"] == method], key=lambda r: tuple(r["selection"])
        )
        ranking.append({"method": method, "median_selection": group[len(group) // 2]["selection"]})
    selected_method = max(ranking, key=lambda r: tuple(r["median_selection"]))["method"]
    selected = next(
        r
        for r in results
        if r["method"] == selected_method and r["seed"] == config["head_seeds"][0]
    )
    write(
        args.output / "selection.json",
        {"selected": selected, "method_ranking": ranking, "final_or_calibration_used": False},
    )
    print(
        json.dumps({"selected": selected["checkpoint"], "selection": selected["selection"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
