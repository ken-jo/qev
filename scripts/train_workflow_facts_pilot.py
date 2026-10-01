"""Teach primitive condition probabilities through an auxiliary training-only decoder."""

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
from veyra.workflow_facts import auxiliary_targets
from veyra.workflow_learning import SHARES, annotate, evaluate, selection_key


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    config_path = Path("configs/workflow-facts-pilot-v12.json")
    config = read(config_path)
    release = read(config["release_protocol"])
    output = Path("runs/workflow-v12-facts-pilot")
    if output.exists():
        raise FileExistsError("pilot experiment is immutable")
    parent_path, features = Path(config["parent"]), Path(config["features"])
    manifest, spec, cache = (
        read(parent_path / "manifest.json"),
        read(features / "specification.json"),
        read(features / "manifest.json"),
    )
    if (
        digest(parent_path / "head.safetensors") != release["baseline_weights_sha256"]
        or digest(parent_path / "manifest.json") != release["baseline_manifest_sha256"]
        or digest(config["records"]) != spec["records_sha256"]
        or spec["calibration_or_final_encoded"] is not False
        or cache["complete"] is not True
    ):
        raise ValueError("frozen parent or training/development corpus mismatch")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(features / name) != cache[key]:
            raise ValueError("feature cache changed")
    tensors = load_file(features / "features.safetensors")
    rows = read(features / "records.json")
    records = [r for r in read_records(Path(config["records"])) if r.split in {"train", "dev"}]
    annotate(rows, records)
    train = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "train"])
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    auxiliary, eligible = auxiliary_targets(rows, records)
    if int(eligible[train].sum()) != 3840 or int(eligible[dev].sum()) != 1200:
        raise ValueError("unexpected auxiliary supervision groups")
    audit_path = Path("data/workflow-v12/conditional-target-audit.json")
    included_groups = {row["group"] for row in rows}
    audit = {
        item["group"]: item["posterior"]
        for item in read(audit_path)
        if item["group"] in included_groups
    }
    for index, row in enumerate(rows):
        if row["domain"] == "uncertainty":
            if not torch.allclose(auxiliary[index], torch.tensor(audit[row["group"]]), atol=1e-6):
                raise ValueError("parsed conditional target disagrees with construction audit")
    state = load_file(parent_path / "head.safetensors")
    temperatures = manifest["calibration"]["temperatures"]
    scale = torch.tensor(temperatures)[tensors["types"], None]
    counts = Counter(rows[i]["domain"] for i in train.tolist())
    weights = torch.tensor(
        [SHARES.get(row["domain"], 0) * len(train) / max(1, counts[row["domain"]]) for row in rows]
    )
    replay = torch.tensor([row["retention"] for row in rows])
    protocol = {
        "config_sha256": digest(config_path),
        "release_protocol_sha256": digest(config["release_protocol"]),
        "scope": config["scope"],
        "feature_specification": spec,
        "matched_control_sha256": digest(config["control"]),
        "source_sha256": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("src/veyra/workflow_facts.py"),
                Path("src/veyra/workflow_learning.py"),
                Path("src/veyra/proper_learning.py"),
            )
        },
        "auxiliary_questions": {
            "train": int(eligible[train].sum()),
            "dev": int(eligible[dev].sum()),
        },
        "conditional_audit_sha256": digest(audit_path),
        "conditional_targets_match_audit": True,
        "domain_shares": SHARES,
        "learning_rates": {"readout": 0.0001, "binding": 0.001},
        "batch_size": 128,
        "epochs": config["epochs"],
        "seeds": config["seeds"],
        "auxiliary_weight": config["auxiliary_weight"],
        "calibration_or_final_used": False,
        "inference_modules_added": 0,
        "primary_experiment_candidates_changed": False,
    }
    write(output / "protocol.json", protocol)
    save_file(
        {"targets": auxiliary, "eligible": eligible}, output / "auxiliary_targets.safetensors"
    )
    torch.set_num_threads(4)
    model = TransferReadout(manifest, state).eval()
    with torch.inference_mode():
        mismatch = float(
            (
                model(tensors["states"][dev], tensors["condition_logits"][dev])
                - tensors["parent_logits"][dev]
            )
            .abs()
            .max()
        )
    if mismatch > 0.0005:
        raise ValueError("initial readout reconstruction mismatch")
    initial = evaluate(model, tensors, rows, dev, temperatures)
    results = [r for r in read(config["control"]) if r["method"] == "rloo"]
    if {r["seed"] for r in results} != set(config["seeds"]):
        raise ValueError("matched controls are incomplete")
    for strength in [config["auxiliary_weight"]]:
        method = f"rloo-facts-{strength}"
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            model = TransferReadout(manifest, state)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 10000)
                decoder = torch.nn.Linear(manifest["binding_head"]["rank"], 4)
            captured = {}

            def capture(_module, _inputs, result):
                captured["projected"] = result

            hook = model.binding_head.project.register_forward_hook(capture)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.readout.parameters(), "lr": 0.0001, "weight_decay": 0.0},
                    {"params": model.binding_head.parameters(), "lr": 0.001, "weight_decay": 0.01},
                    {"params": decoder.parameters(), "lr": 0.001, "weight_decay": 0.01},
                ]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config["epochs"], eta_min=0.00001
            )
            best_key, best_epoch, best_metrics = selection_key(initial, initial), 0, initial
            best_state = copy.deepcopy(model.state_dict())
            best_decoder = copy.deepcopy(decoder.state_dict())
            started, history, steps = time.perf_counter(), [], 0
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                total, agreement_total = 0.0, 0.0
                for batch in grouped_batches(
                    train, rows, 128, torch.Generator().manual_seed(seed * 1000 + epoch)
                ):
                    logits = (
                        model(tensors["states"][batch], tensors["condition_logits"][batch])
                        / scale[batch]
                    )
                    valid = tensors["valid"][batch]
                    losses = distribution_losses(
                        logits, tensors["targets"][batch], valid, tensors["types"][batch], "rloo"
                    )
                    losses = losses * torch.where(replay[batch], 0.25, 1.0)
                    losses += (
                        4
                        * retention_losses(
                            logits, tensors["parent_logits"][batch] / scale[batch], valid
                        )
                        * replay[batch]
                    )
                    predictions = decoder(torch.nn.functional.gelu(captured["projected"]))
                    agreement = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            predictions, auxiliary[batch], reduction="none"
                        ).mean(-1)
                        * eligible[batch]
                    )
                    loss = ((losses + strength * agreement) * weights[batch]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite auxiliary objective")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(decoder.parameters()),
                        1.0,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    steps += 1
                    total += float(loss.detach()) * len(batch)
                    agreement_total += float(agreement.detach().sum())
                scheduler.step()
                metrics = evaluate(model, tensors, rows, dev, temperatures)
                with torch.inference_mode():
                    probabilities = decoder(
                        torch.nn.functional.gelu(captured["projected"])
                    ).sigmoid()
                    auxiliary_mse = float(
                        (probabilities[eligible[dev]] - auxiliary[dev][eligible[dev]])
                        .square()
                        .mean()
                    )
                key = selection_key(metrics, initial)
                if key > best_key:
                    best_key, best_epoch, best_metrics = key, epoch, metrics
                    best_state = copy.deepcopy(model.state_dict())
                    best_decoder = copy.deepcopy(decoder.state_dict())
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": total / len(train),
                        "train_auxiliary_bce": agreement_total / len(train),
                        "dev_auxiliary_probability_mse": auxiliary_mse,
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
                write(output / "progress.json", progress)
                if epoch % 5 == 0:
                    print(json.dumps(progress), flush=True)
            folder = output / f"{method}-{seed}"
            checkpoint = folder / "checkpoint"
            checkpoint.mkdir(parents=True)
            weights_state = dict(state)
            weights_state.update({key: value.contiguous() for key, value in best_state.items()})
            save_file(weights_state, checkpoint / "head.safetensors")
            save_file(best_decoder, folder / "auxiliary-training-only.safetensors")
            saved = copy.deepcopy(manifest)
            saved["weights_sha256"] = digest(checkpoint / "head.safetensors")
            saved["training"] = {
                "completed": True,
                "intermediate": True,
                "optimizer_steps": steps,
                "method": method,
                "seed": seed,
                "selected_epoch": best_epoch,
                "selected_dev": best_metrics,
                "selection": best_key,
                "parent_training": manifest["training"],
                "dataset_sha256": spec["records_sha256"],
                "protocol_sha256": digest(output / "protocol.json"),
                "final_evaluated_at_selection": False,
                "exploratory_pilot": True,
                "auxiliary_decoder_in_checkpoint": False,
                "smoke": False,
            }
            write(checkpoint / "manifest.json", saved)
            write(folder / "history.json", history)
            results.append(
                {
                    "method": method,
                    "seed": seed,
                    "selected_epoch": best_epoch,
                    "selection": best_key,
                    "dev": best_metrics,
                    "checkpoint": str(checkpoint),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            write(output / "results.json", results)
            hook.remove()
    ranking = []
    for method in sorted({r["method"] for r in results}):
        group = sorted(
            [r for r in results if r["method"] == method], key=lambda r: tuple(r["selection"])
        )
        ranking.append({"method": method, "median_selection": group[len(group) // 2]["selection"]})
    winner = max(ranking, key=lambda row: tuple(row["median_selection"]))["method"]
    selected = next(r for r in results if r["method"] == winner and r["seed"] == config["seeds"][0])
    write(
        output / "comparison.json",
        {
            "selected": selected,
            "ranking": ranking,
            "calibration_or_final_used": False,
            "primary_experiment_candidates_changed": False,
            "release_allowed": False,
            "requires_new_declared_evaluation_cycle": True,
        },
    )
    write(
        output / "complete.json",
        {"completed": True, "selected_method": winner, "release_allowed": False},
    )
    print(json.dumps({"completed": True, "selected_method": winner}), flush=True)


if __name__ == "__main__":
    main()
