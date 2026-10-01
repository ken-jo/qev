"""Compare hard-mode and soft teacher targets on the frozen train/development corpus."""

import copy
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from train_foundation_head import write
from train_workflow_workspace import development_risk
from workspace_development_diagnostics import diagnose

from veyra.data import read_records
from veyra.proper_learning import distribution_losses, grouped_batches, retention_losses
from veyra.transfer_learning import TransferReadout
from veyra.workflow_learning import SHARES, annotate, evaluate, selection_key


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    config_path = Path("configs/workflow-mode-target-pilot-v12.json")
    config = read(config_path)
    if config["status"] != "declared_before_training":
        raise ValueError("declare the target-semantics pilot before training")
    release = read(config["release_protocol"])
    if (
        digest(config["control"]) != config["control_sha256"]
        or digest("reports/workflow-v12/training-target-audit.json")
        != config["training_target_audit_sha256"]
    ):
        raise ValueError("declared control or target audit changed")
    output = Path("runs/workflow-v12-mode-target-pilot")
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
    teacher = torch.tensor(
        [row["split"] == "train" and "teacher_distribution" in row["tags"] for row in rows]
    )
    if int(teacher.sum()) != 4200 or (tensors["gold"][teacher] < 0).any():
        raise ValueError("unexpected teacher-mode supervision population")
    train = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "train"])
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    state = load_file(parent_path / "head.safetensors")
    temperatures = manifest["calibration"]["temperatures"]
    scale = torch.tensor(temperatures)[tensors["types"], None]
    counts = Counter(rows[i]["domain"] for i in train.tolist())
    weights = torch.tensor(
        [SHARES.get(row["domain"], 0) * len(train) / max(1, counts[row["domain"]]) for row in rows]
    )
    replay = torch.tensor([row["retention"] for row in rows])
    protocol = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(config_path),
        "release_protocol_sha256": digest(config["release_protocol"]),
        "scope": config["scope"],
        "feature_specification": spec,
        "matched_control_sha256": digest(config["control"]),
        "source_sha256": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("scripts/workspace_development_diagnostics.py"),
                Path("src/veyra/workflow_learning.py"),
                Path("src/veyra/proper_learning.py"),
            )
        },
        "training_teacher_mode_questions": int(teacher.sum()),
        "domain_shares": SHARES,
        "learning_rates": {"readout": 0.0001, "binding": 0.001},
        "batch_size": 128,
        "epochs": config["epochs"],
        "seeds": config["seeds"],
        "hard_mode_weights": config["hard_mode_weights"],
        "calibration_or_final_used": False,
        "inference_modules_added": 0,
        "primary_experiment_candidates_changed": False,
    }
    write(output / "protocol.json", protocol)
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

    def diagnostic(checkpoint):
        checkpoint = Path(checkpoint)
        saved = read(checkpoint / "manifest.json")
        saved_state = load_file(checkpoint / "head.safetensors")
        if any(
            not torch.equal(value, state[key])
            for key, value in saved_state.items()
            if key != "readout.weight" and not key.startswith("binding_head.")
        ):
            raise ValueError("diagnostic candidate changed the frozen backbone")
        candidate = TransferReadout(saved, saved_state).eval()
        with torch.inference_mode():
            logits = candidate(tensors["states"][dev], tensors["condition_logits"][dev])
        return {
            "risk": development_risk(logits, tensors, dev, temperatures),
            "additional": diagnose(
                logits, tensors, rows, dev, {r.id: r for r in records}, temperatures
            ),
        }

    for result in results:
        result["diagnostics"] = diagnostic(result["checkpoint"])
    for strength in config["hard_mode_weights"]:
        method = f"rloo-mode-{strength}"
        training_targets = tensors["targets"].clone()
        hard = torch.nn.functional.one_hot(tensors["gold"][teacher], num_classes=16).float()
        training_targets[teacher] = (1 - strength) * training_targets[teacher] + strength * hard
        if not torch.equal(training_targets[~teacher], tensors["targets"][~teacher]):
            raise ValueError("non-teacher or held-out targets changed")
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            model = TransferReadout(manifest, state)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.readout.parameters(), "lr": 0.0001, "weight_decay": 0.0},
                    {"params": model.binding_head.parameters(), "lr": 0.001, "weight_decay": 0.01},
                ]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config["epochs"], eta_min=0.00001
            )
            best_key, best_epoch, best_metrics = selection_key(initial, initial), 0, initial
            best_state = copy.deepcopy(model.state_dict())
            started, history, steps = time.perf_counter(), [], 0
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                total = 0.0
                for batch in grouped_batches(
                    train, rows, 128, torch.Generator().manual_seed(seed * 1000 + epoch)
                ):
                    logits = (
                        model(tensors["states"][batch], tensors["condition_logits"][batch])
                        / scale[batch]
                    )
                    valid = tensors["valid"][batch]
                    losses = distribution_losses(
                        logits, training_targets[batch], valid, tensors["types"][batch], "rloo"
                    )
                    losses = losses * torch.where(replay[batch], 0.25, 1.0)
                    losses += (
                        4
                        * retention_losses(
                            logits, tensors["parent_logits"][batch] / scale[batch], valid
                        )
                        * replay[batch]
                    )
                    loss = (losses * weights[batch]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite teacher-mode objective")
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
                        "teacher_mode_weight": strength,
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
                    "diagnostics": diagnostic(checkpoint),
                }
            )
            write(output / "results.json", results)
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
