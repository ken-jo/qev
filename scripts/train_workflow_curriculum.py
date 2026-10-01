"""Matched backbone continuation with direct prerequisite queries or outcome replay."""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_foundation_head import write
from train_workflow_backbone import forward
from train_workflow_workspace import development_risk
from workflow_skill_common import assess_skills, canonical_target

from veyra.data import TrainingRecord, read_records
from veyra.option_model import OptionModel
from veyra.proper_learning import distribution_losses, retention_losses
from veyra.workflow_learning import SHARES, annotate, selection_key, summarize_logits


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("skill_curriculum", "outcome_replay"), required=True)
    args = parser.parse_args()
    config_path = Path("configs/workflow-curriculum-study-v12.json")
    config = read(config_path)
    output = Path("runs/workflow-v12-curriculum-study") / args.arm
    if output.exists():
        raise FileExistsError("curriculum arm outputs are immutable")
    if config["status"] != "declared_before_training" or args.arm not in config["arms"]:
        raise ValueError("undeclared training arm")
    parent, data, features, skills = map(
        Path, (config["parent"], config["records"], config["features"], config["skills"])
    )
    bindings = {
        parent / "head.safetensors": config["parent_weights_sha256"],
        parent / "manifest.json": config["parent_manifest_sha256"],
        data: config["records_sha256"],
        features / "manifest.json": config["feature_cache_manifest_sha256"],
        skills / "audit.json": config["skills_audit_sha256"],
        skills / "independent-target-audit.json": config["skills_independent_audit_sha256"],
        Path(config["training_plan"]): config["training_plan_sha256"],
        Path(config["release_protocol"]): config["release_protocol_sha256"],
        Path(config["baseline_development"]): config["baseline_development_sha256"],
    }
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError("declared training input changed: " + str(path))
    cache = read(features / "manifest.json")
    spec = read(features / "specification.json")
    if spec["records_sha256"] != digest(data) or spec["calibration_or_final_encoded"] is not False:
        raise ValueError("unexpected feature cache source")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(features / name) != cache[key]:
            raise ValueError("feature cache changed")
    for name, expected in read(skills / "audit.json")["files_sha256"].items():
        if digest(skills / name) != expected:
            raise ValueError("skill data changed")
    records = {}
    for line in data.open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] in {"train", "dev"}:
            record = TrainingRecord.model_validate(raw)
            records[record.id] = record
    skill_train = {r.id: r for r in read_records(skills / "train.jsonl")}
    skill_dev = {r.id: r for r in read_records(skills / "dev.jsonl")}
    diagnostic = [skill_dev[key] for key in read(skills / "diagnostic-ids.json")]
    if any(r.split != "train" for r in skill_train.values()) or any(
        r.split != "dev" for r in diagnostic
    ):
        raise ValueError("skill data roles overlap")
    rows, tensors = read(features / "records.json"), load_file(features / "features.safetensors")
    annotate(rows, records.values())
    lookup = {row["id"]: i for i, row in enumerate(rows)}
    train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    counts = Counter(rows[i]["domain"] for i in train)
    weights = {domain: SHARES[domain] * len(train) / count for domain, count in counts.items()}
    orders = read(config["training_plan"])["epochs"]
    if len(orders) != config["epochs"]:
        raise ValueError("epoch plan changed")
    for order in orders:
        if Counter(item["extra"] for item in order) != {False: 11212, True: 3840}:
            raise ValueError("training budget changed")
        for item in order:
            if item["extra"]:
                control = records[item["control_id"]]
                skill = skill_train[item["skill_id"]]
                if (
                    control.split != "train"
                    or skill.split != "train"
                    or control.group_id != skill.group_id
                    or control.request.state != skill.request.state
                ):
                    raise ValueError("invalid paired skill/control query")
            elif (
                rows[item["cache_index"]]["id"] != item["source_id"]
                or records[item["source_id"]].split != "train"
            ):
                raise ValueError("main schedule no longer matches the training cache")
    targets = {key: canonical_target(record) for key, record in skill_train.items()}
    source_files = [
        Path(__file__),
        Path("scripts/workflow_skill_common.py"),
        Path("scripts/train_workflow_backbone.py"),
        Path("scripts/train_workflow_workspace.py"),
        Path("src/veyra/option_model.py"),
        Path("src/veyra/proper_learning.py"),
        Path("src/veyra/workflow_learning.py"),
    ]
    source_hashes = {str(path): digest(path) for path in source_files}
    write(
        output / "protocol.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": digest(config_path),
            "arm": args.arm,
            "source_sha256": source_hashes,
            "feature_cache": cache,
            "training_plan_sha256": config["training_plan_sha256"],
            "skills_audit_sha256": config["skills_audit_sha256"],
            "calibration_or_final_used": False,
            "inference_modules_added": 0,
        },
    )
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    model = OptionModel.load(parent, local_files_only=True, merge=False)
    model.condition_readout.requires_grad_(False)
    groups = [
        {
            "params": [p for name, p in model.named_parameters() if ".lora_" in name],
            "lr": config["learning_rate"],
        },
        {
            "params": list(model.readout.parameters()) + list(model.binding_head.parameters()),
            "lr": config["learning_rate"] * config["readout_lr_multiplier"],
        },
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    accumulation = config["accumulation"]
    total_steps = sum(math.ceil(len(order) / accumulation) for order in orders)

    def rate(step):
        if step < 30:
            return (step + 1) / 30
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1, (step - 30) / max(1, total_steps - 30))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    temperatures = model.calibration.temperatures
    baseline = read(config["baseline_development"])["metrics"]
    steps = processed = 0
    loss_total = 0.0
    history = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch, order in enumerate(orders, 1):
        model.train()
        model.encoder.model.eval()
        for start in range(0, len(order), accumulation):
            batch = order[start : start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for item in batch:
                use_skill = item["extra"] and args.arm == "skill_curriculum"
                if use_skill:
                    record = skill_train[item["skill_id"]]
                    target, valid, type_id = targets[record.id]
                    target, valid = target[None], valid[None]
                    types = torch.tensor([type_id])
                else:
                    identifier = item["control_id"] if item["extra"] else item["source_id"]
                    i = lookup[identifier]
                    record = records[identifier]
                    target, valid, types = (
                        tensors[key][i : i + 1] for key in ("targets", "valid", "types")
                    )
                logits, hidden = forward(model, record, data.parent)
                target, valid, types = (value.to(logits.device) for value in (target, valid, types))
                scale = torch.tensor(temperatures, device=logits.device)[types, None]
                scaled = logits / scale
                loss = distribution_losses(scaled, target, valid, types, config["method"]).mean()
                if not item["extra"]:
                    if rows[i]["retention"]:
                        reference = tensors["parent_logits"][i : i + 1].to(logits.device) / scale
                        anchor = (
                            1
                            - torch.nn.functional.cosine_similarity(
                                hidden, tensors["states"][i : i + 1].to(hidden.device)
                            ).mean()
                        )
                        loss = (
                            0.25 * loss
                            + 4 * retention_losses(scaled, reference, valid).mean()
                            + 0.1 * anchor
                        )
                    loss = loss * weights[rows[i]["domain"]]
                else:
                    loss = loss * config["extra_loss_weight"]
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite curriculum objective")
                (loss / len(batch)).backward()
                processed += 1
                loss_total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            steps += 1
            if steps % 25 == 0:
                progress = {
                    "arm": args.arm,
                    "epoch": epoch,
                    "processed": processed,
                    "total": sum(map(len, orders)),
                    "steps": steps,
                    "mean_loss": loss_total / processed,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                write(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        model.eval()
        with torch.inference_mode():
            # Keep the declared original confidence screen independent of skill-query results.
            logits = torch.stack(
                [
                    forward(model, records[rows[i]["id"]], data.parent)[0][0].cpu()
                    for i in dev.tolist()
                ]
            )
        metrics = summarize_logits(logits, tensors, rows, dev, temperatures)
        risk = development_risk(logits, tensors, dev, temperatures)
        skill_metrics, _ = assess_skills(model, diagnostic, skills)
        checkpoint = output / f"epoch-{epoch}" / "checkpoint"
        for path, expected in source_hashes.items():
            if digest(path) != expected:
                raise ValueError("source changed during training")
        metadata = {
            "completed": True,
            "intermediate": True,
            "method": "curriculum-" + args.arm,
            "seed": config["seed"],
            "epochs": epoch,
            "optimizer_steps": steps,
            "backbone_updated": True,
            "parent_weights_sha256": config["parent_weights_sha256"],
            "dataset_sha256": config["records_sha256"],
            "skills_audit_sha256": config["skills_audit_sha256"],
            "protocol_sha256": digest(output / "protocol.json"),
            "selected_dev": metrics,
            "selection": selection_key(metrics, baseline),
            "development_risk_diagnostic": risk,
            "skill_diagnostic": skill_metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "final_evaluated_at_selection": False,
            "smoke": False,
        }
        model.save(checkpoint, metadata)
        # Preserve local optimizer/RNG state for a separately audited recovery if interrupted.
        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "steps": steps,
                "processed": processed,
                "loss_total": loss_total,
                "protocol_sha256": metadata["protocol_sha256"],
            },
            checkpoint.parent / "optimizer-local.pt",
        )
        history.append({"checkpoint": str(checkpoint), **metadata})
        write(output / "history.json", history)
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "selection": metadata["selection"],
                    "risk": risk,
                    "skills": skill_metrics,
                }
            ),
            flush=True,
        )
    write(
        output / "complete.json",
        {
            "completed": True,
            "candidates": [item["checkpoint"] for item in history],
            "release_allowed": False,
        },
    )


if __name__ == "__main__":
    main()
