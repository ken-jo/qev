"""Declared head interpolation and replay; CPU train/dev only, no final inference."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from train_foundation_head import write
from train_workflow_workspace import development_risk

from veyra.transfer_learning import TransferReadout, transfer_metrics
from veyra.workflow_learning import selection_key, summarize_logits

CONFIG = Path("configs/workflow-readout-recovery-v13.json")
CACHE = Path("data/features-readout-recovery-v13")
OUTPUT = Path("runs/readout-recovery-v13")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache(split):
    manifest = read(CACHE / split / "manifest.json")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
    ):
        if digest(CACHE / split / name) != manifest[key]:
            raise ValueError("Feature cache changed")
    if (
        not manifest["complete"]
        or manifest["split"] != split
        or digest(CACHE / "specification.json") != manifest["specification_sha256"]
    ):
        raise ValueError("Incomplete cache")
    spec = read(CACHE / "specification.json")
    if spec["design_sha256"] != digest(CONFIG) or spec["calibration_or_final_encoded"]:
        raise ValueError("Wrong cache design or prohibited split")
    for path, expected in spec["source_files"].items():
        if digest(path) != expected:
            raise ValueError("Cached source changed: " + path)
    rows = read(CACHE / split / "records.json")
    if any(row["split"] != split for row in rows):
        raise ValueError("Wrong split in cache")
    return load_file(CACHE / split / "features.safetensors"), rows


@torch.inference_mode()
def assess(model, tensors, rows, temperatures):
    model.eval()
    logits = model(tensors["states"], tensors["condition_logits"])
    workflow = torch.tensor([i for i, r in enumerate(rows) if r["corpus"] == "workflow"])
    metrics = summarize_logits(logits[workflow], tensors, rows, workflow, temperatures)
    risk = development_risk(logits[workflow], tensors, workflow, temperatures)
    legacy = {}
    for value in ("text", "image"):
        indices = torch.tensor(
            [i for i, r in enumerate(rows) if r["corpus"] == "legacy" and r["modality"] == value]
        )
        legacy[value] = transfer_metrics(
            logits[indices],
            tensors["targets"][indices],
            tensors["valid"][indices],
            tensors["gold"][indices],
            tensors["types"][indices],
        )
    return {"workflow": metrics, "legacy": legacy, "development_risk": risk}, logits


def decide(result, parent, baseline, config, movement):
    limits = config["selection"]
    domains, old = result["workflow"]["by_domain"], parent["workflow"]["by_domain"]
    unc, old_unc = domains["uncertainty"], old["uncertainty"]
    primary = selection_key(result["workflow"], baseline)
    checks = {
        "original_development_gates": bool(primary[0]),
        "legacy_text": result["legacy"]["text"]["accuracy"] >= limits["legacy_text_accuracy_min"],
        "legacy_image": result["legacy"]["image"]["accuracy"]
        >= limits["legacy_image_accuracy_min"],
        "workflow_retained": all(
            domains[d]["accuracy"]
            >= old[d]["accuracy"]
            - limits["workflow_known_and_new_accuracy_drop_max_from_parent"]
            - 1e-9
            for d in ("workflow_known", "workflow_new")
        ),
        "uncertainty_nll_retained": unc["nll"]
        <= old_unc["nll"] + limits["uncertainty_nll_max_increase_from_parent"],
        "uncertainty_brier_retained": unc["brier_vs_soft"]
        <= old_unc["brier_vs_soft"] + limits["uncertainty_brier_max_increase_from_parent"],
        "uncertainty_cost_retained": unc["matched_coverage"]["0.8"]["expected_cost"]
        <= old_unc["matched_coverage"]["0.8"]["expected_cost"]
        + limits["uncertainty_cost80_max_increase_from_parent"],
        "development_policy_feasible": all(
            r["maximum_coverage_at_15pct_error"] >= 0.6 for r in result["development_risk"].values()
        ),
    }
    rank = [
        all(checks.values()),
        min(r["accuracy"] for r in result["legacy"].values()),
        (domains["workflow_known"]["accuracy"] + domains["workflow_new"]["accuracy"]) / 2,
        -unc["nll"],
        -movement,
    ]
    return checks, rank


def save_candidate(name, model, parent_manifest, parent_state, metadata):
    folder = OUTPUT / name
    if folder.exists():
        raise FileExistsError("Candidate is immutable: " + name)
    checkpoint = folder / "checkpoint"
    checkpoint.mkdir(parents=True)
    state = {key: value.clone() for key, value in parent_state.items()}
    state.update(
        {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    )
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Nonfinite candidate")
    for key in parent_state:
        if key != "readout.weight" and not key.startswith("binding_head."):
            if not torch.equal(state[key], parent_state[key]):
                raise ValueError("Frozen parameter changed: " + key)
    save_file(state, checkpoint / "head.safetensors")
    manifest = copy.deepcopy(parent_manifest)
    old_training = manifest["training"]
    manifest["training"] = {
        "completed": True,
        "intermediate": False,
        "optimizer_steps": old_training["optimizer_steps"] + metadata["additional_steps"],
        "parent_manifest_sha256": metadata["parent_manifest_sha256"],
        "readout_recovery": metadata,
        "release_allowed": False,
        "calibration_status": "Parent calibration retained for development only; refit required",
    }
    manifest["weights_sha256"] = digest(checkpoint / "head.safetensors")
    write(checkpoint / "manifest.json", manifest)
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("interpolation", "training"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = read(CONFIG)
    for role in ("parent", "foundation"):
        for name, file in (("weights", "head.safetensors"), ("manifest", "manifest.json")):
            if digest(Path(config[role]) / file) != config[role + "_" + name + "_sha256"]:
                raise ValueError("Declared model changed")
    phase_file = OUTPUT / (args.phase + "-protocol.json")
    if phase_file.exists():
        raise FileExistsError("Study phase is immutable")
    tensors, rows = cache("dev")
    parent_manifest = read(Path(config["parent"]) / "manifest.json")
    parent_state = load_file(Path(config["parent"]) / "head.safetensors")
    parent_model = TransferReadout(parent_manifest, parent_state).eval()
    temps = parent_manifest["calibration"]["temperatures"]
    parent_metrics, parent_logits = assess(parent_model, tensors, rows, temps)
    maximum_error = float((parent_logits - tensors["parent_logits"]).abs().max())
    if maximum_error > 0.0005:
        raise ValueError("CPU readout does not reconstruct current GPU outputs")
    baseline_path = Path("runs/workflow-v12-workspace-selection/merged-baseline-dev.json")
    baseline = read(baseline_path)["metrics"]
    protocol = {
        "design_sha256": digest(CONFIG),
        "source_sha256": digest(Path(__file__)),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "cpu_parent_maximum_logit_error": maximum_error,
        "development_cache_manifest_sha256": digest(CACHE / "dev/manifest.json"),
        "foundation_development_sha256": digest(baseline_path),
        "parent_development": parent_metrics,
        "calibration_or_final_used": False,
        "release_allowed": False,
    }
    if args.phase == "training":
        train, train_rows = cache("train")
        if {r["group"] for r in train_rows} & {r["group"] for r in rows}:
            raise ValueError("Training/development overlap")
        protocol["training_cache_manifest_sha256"] = digest(CACHE / "train/manifest.json")
    write(phase_file, protocol)
    results = []

    def report(name, model, additional_steps, **extra):
        metrics, _ = assess(model, tensors, rows, temps)
        movement = sum(
            float((value - parent_state[key]).square().sum())
            for key, value in model.state_dict().items()
        )
        checks, rank = decide(metrics, parent_metrics, baseline, config, movement)
        metadata = {
            "experiment": config["experiment"],
            "design_sha256": digest(CONFIG),
            "phase_protocol": str(phase_file),
            "phase_protocol_sha256": digest(phase_file),
            "parent_manifest_sha256": config["parent_manifest_sha256"],
            "parent_weights_sha256": config["parent_weights_sha256"],
            "additional_steps": additional_steps,
            "frozen_backbone": True,
            **extra,
        }
        checkpoint = save_candidate(name, model, parent_manifest, parent_state, metadata)
        result = {
            "name": name,
            "checkpoint": str(checkpoint),
            "metrics": metrics,
            "checks": checks,
            "selection": rank,
            "squared_parameter_movement": movement,
            "weights_sha256": digest(checkpoint / "head.safetensors"),
            "manifest_sha256": digest(checkpoint / "manifest.json"),
        }
        write(OUTPUT / name / "development.json", result)
        results.append(result)
        print(
            json.dumps(
                {
                    "candidate": name,
                    "checks": checks,
                    "rank": rank,
                    "legacy": {k: v["accuracy"] for k, v in metrics["legacy"].items()},
                }
            ),
            flush=True,
        )

    if args.phase == "interpolation":
        foundation_state = load_file(Path(config["foundation"]) / "head.safetensors")
        for fraction in config["head_interpolation_foundation_fractions"]:
            model = copy.deepcopy(parent_model)
            mixed = {
                key: parent_state[key] * (1 - fraction) + foundation_state[key] * fraction
                for key in model.state_dict()
            }
            model.load_state_dict(mixed, strict=True)
            report(f"interpolation-{fraction:g}", model, 0, foundation_fraction=fraction)
    else:
        settings = config["training"]
        torch.manual_seed(config["seed"])
        model = copy.deepcopy(parent_model)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.readout.parameters(), "lr": settings["learning_rates"]["readout"]},
                {
                    "params": model.binding_head.parameters(),
                    "lr": settings["learning_rates"]["binding"],
                },
            ],
            weight_decay=settings["weight_decay"],
        )
        families = {
            corpus: {r["family"] for r in train_rows if r["corpus"] == corpus}
            for corpus in ("workflow", "legacy")
        }
        counts = Counter((r["corpus"], r["family"]) for r in train_rows)
        weights = torch.tensor(
            [
                settings[r["corpus"] + "_total_loss_share"]
                / (len(families[r["corpus"]]) * counts[(r["corpus"], r["family"])])
                for r in train_rows
            ]
        ) * len(train_rows)
        anchor = torch.tensor([r["corpus"] == "workflow" for r in train_rows])
        scale = torch.tensor(temps)[train["types"], None]
        steps = 0
        for epoch in range(1, settings["epochs"] + 1):
            model.train()
            order = torch.randperm(len(train_rows))
            total_loss = 0.0
            for indices in order.split(settings["batch_size"]):
                logits = model(train["states"][indices], train["condition_logits"][indices])
                valid = train["valid"][indices]
                logp = (logits / scale[indices]).masked_fill(~valid, -1e9).log_softmax(-1)
                ce = -(train["targets"][indices] * logp).sum(-1)
                teacher = (
                    (train["parent_logits"][indices] / 2).masked_fill(~valid, -1e9).softmax(-1)
                )
                student = (logits / 2).masked_fill(~valid, -1e9).log_softmax(-1)
                kl = (teacher * (teacher.clamp_min(1e-12).log() - student)).sum(-1)
                loss = ((ce + 4 * kl * anchor[indices]) * weights[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite replay loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"])
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
                steps += 1
            write(
                OUTPUT / "training-progress.json",
                {
                    "epoch": epoch,
                    "steps": steps,
                    "mean_loss": total_loss / len(train_rows),
                    "release_allowed": False,
                },
            )
            if epoch in settings["candidate_epochs"]:
                report(f"replay-epoch-{epoch}", model, steps, epoch=epoch, seed=config["seed"])
    write(
        OUTPUT / (args.phase + "-complete.json"),
        {
            "completed": True,
            "candidates": results,
            "source_sha256": digest(Path(__file__)),
            "release_allowed": False,
        },
    )
    if args.phase == "training":
        all_results = read(OUTPUT / "interpolation-complete.json")["candidates"] + results
        selected = max(all_results, key=lambda result: tuple(result["selection"]))
        write(
            OUTPUT / "selection.json",
            {
                "selected": selected,
                "eligible": selected["selection"][0],
                "design_sha256": digest(CONFIG),
                "candidates": [r["name"] for r in all_results],
                "requires_actual_merged_gpu_verification": True,
                "final_or_calibration_used": False,
                "release_allowed": False,
            },
        )


if __name__ == "__main__":
    main()
