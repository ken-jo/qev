"""Matched continuation study with optional primitive-fact gradients into the backbone."""

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from train_foundation_head import write
from train_workflow_backbone import forward
from workflow_workspace_supervision import WorkspaceSupervisor

from veyra.constants import QUESTION_TYPES
from veyra.data import read_records
from veyra.option_model import OptionModel
from veyra.proper_learning import distribution_losses, retention_losses
from veyra.workflow_facts import auxiliary_targets
from veyra.workflow_learning import SHARES, annotate, selection_key, summarize_logits


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def development_risk(logits, tensors, indices, temperatures):
    types, valid = tensors["types"][indices], tensors["valid"][indices]
    values = logits / torch.tensor(temperatures)[types, None]
    confidence, prediction = values.masked_fill(~valid, -1e9).softmax(-1).max(-1)
    gold = tensors["gold"][indices]
    errors = 1 - tensors["targets"][indices].gather(1, prediction[:, None]).squeeze(1)
    errors = torch.where(gold >= 0, (prediction != gold).float(), errors)
    result = {}
    for type_id, kind in enumerate(QUESTION_TYPES):
        selected = torch.where(types == type_id)[0]
        order = selected[torch.argsort(confidence[selected], descending=True, stable=True)]
        count = len(order)
        counts = torch.arange(1, count + 1)
        risk = errors[order].cumsum(0) / counts
        boundaries = torch.ones(count, dtype=torch.bool)
        boundaries[:-1] = confidence[order[:-1]] != confidence[order[1:]]
        eligible = torch.where((risk <= 0.15) & (counts >= 20) & boundaries)[0]
        result[kind] = {
            "questions": count,
            "risk_at_60pct_coverage": float(risk[max(0, math.floor(count * 0.6) - 1)]),
            "maximum_coverage_at_15pct_error": (int(eligible[-1]) + 1) / count
            if len(eligible)
            else 0.0,
        }
    return result


@torch.inference_mode()
def assess(model, supervisor, enabled, records, root, rows, tensors, indices, targets, eligible):
    model.eval()
    logits, fact_prediction, fact_target = [], [], []
    for i in indices.tolist():
        with supervisor.observe(enabled and bool(eligible[i])):
            values, _ = forward(model, records[rows[i]["id"]], root)
            logits.append(values[0].cpu())
            if enabled and eligible[i]:
                fact_prediction.append(supervisor.logits()[0].sigmoid().cpu())
                fact_target.append(targets[i])
    logits = torch.stack(logits)
    metrics = summarize_logits(logits, tensors, rows, indices, model.calibration.temperatures)
    result = {
        "metrics": metrics,
        "development_risk_diagnostic": development_risk(
            logits, tensors, indices, model.calibration.temperatures
        ),
    }
    if fact_prediction:
        prediction, target = torch.stack(fact_prediction), torch.stack(fact_target)
        observed = (target == 0) | (target == 1)
        result["auxiliary_diagnostic"] = {
            "questions": len(prediction),
            "condition_probability_mse": float((prediction - target).square().mean()),
            "observed_condition_accuracy": float(
                ((prediction[observed] >= 0.5) == (target[observed] == 1)).float().mean()
            ),
            "constant_half_mse": float((target - 0.5).square().mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("workspace study outputs are immutable")
    config = read(args.config)
    if config["status"] != "declared_before_training":
        raise ValueError("declare the follow-up experiment before optimization")
    arm = config["arms"][args.arm]
    parent, data, features = map(Path, (config["parent"], config["records"], config["features"]))
    if (
        digest(parent / "head.safetensors") != config["parent_weights_sha256"]
        or digest(parent / "manifest.json") != config["parent_manifest_sha256"]
        or digest(data) != config["records_sha256"]
        or digest(config["release_protocol"]) != config["release_protocol_sha256"]
        or digest(config["baseline_development"]) != config["baseline_development_sha256"]
    ):
        raise ValueError("declared model, data or acceptance protocol changed")
    spec, cache = read(features / "specification.json"), read(features / "manifest.json")
    if spec["records_sha256"] != digest(data) or spec["calibration_or_final_encoded"] is not False:
        raise ValueError("feature cache is not the declared training/development set")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(features / name) != cache[key]:
            raise ValueError("feature cache bytes changed")
    tensors, rows = load_file(features / "features.safetensors"), read(features / "records.json")
    records = {r.id: r for r in read_records(data) if r.split in {"train", "dev"}}
    annotate(rows, records.values())
    targets, eligible = auxiliary_targets(rows, records.values())
    train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    if int(eligible[train].sum()) != 3840 or int(eligible[dev].sum()) != 1200:
        raise ValueError("unexpected primitive supervision population")
    conditional_audit = Path("data/workflow-v12/conditional-target-audit.json")
    included_groups = {row["group"] for row in rows}
    audited_targets = {
        item["group"]: item["posterior"]
        for item in read(conditional_audit)
        if item["group"] in included_groups
    }
    for i, row in enumerate(rows):
        if row["domain"] == "uncertainty" and not torch.allclose(
            targets[i], torch.tensor(audited_targets[row["group"]]), atol=1e-6
        ):
            raise ValueError("primitive target differs from the original evidence model")
    counts = Counter(rows[i]["domain"] for i in train)
    weights = {domain: SHARES[domain] * len(train) / count for domain, count in counts.items()}
    grouped = {}
    for i in train:
        grouped.setdefault(rows[i]["group"], []).append(i)
    orders = []
    for epoch in range(config["epochs"]):
        groups = list(grouped.values())
        random.Random(config["seed"] + epoch).shuffle(groups)
        orders.append([i for group in groups for i in group])
    baseline = read(config["baseline_development"])["dev"]
    protocol = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "config_sha256": digest(args.config),
        "arm": args.arm,
        "source_sha256": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("scripts/workflow_workspace_supervision.py"),
                Path("scripts/train_workflow_backbone.py"),
                Path("src/veyra/workflow_facts.py"),
                Path("src/veyra/workflow_learning.py"),
                Path("src/veyra/proper_learning.py"),
                Path("src/veyra/option_model.py"),
            )
        },
        "feature_cache": cache,
        "conditional_audit_sha256": digest(conditional_audit),
        "conditional_targets_match_audit": True,
        "domain_shares": SHARES,
        "calibration_or_final_used": False,
        "primary_experiment_candidates_changed": False,
        "inference_modules_added": 0,
        "generated_intermediate_tokens": 0,
    }
    write(args.output / "protocol.json", protocol)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    model = OptionModel.load(parent, local_files_only=True, merge=False)
    model.condition_readout.requires_grad_(False)
    supervisor = WorkspaceSupervisor(model, config["seed"])
    adapters = [p for name, p in model.named_parameters() if ".lora_" in name]
    groups = [
        {"params": adapters, "lr": config["learning_rate"]},
        {
            "params": list(model.readout.parameters()) + list(model.binding_head.parameters()),
            "lr": config["learning_rate"] * 0.1,
        },
    ]
    if arm["auxiliary_weight"] > 0:
        groups.append({"params": supervisor.decoder.parameters(), "lr": config["auxiliary_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    accumulation = config["accumulation"]
    total_steps = len(orders) * math.ceil(len(train) / accumulation)

    def rate(step):
        if step < 30:
            return (step + 1) / 30
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1, (step - 30) / max(1, total_steps - 30))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    temperatures = model.calibration.temperatures
    history, steps, processed, total_loss, auxiliary_count, auxiliary_sum = [], 0, 0, 0.0, 0, 0.0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch, order in enumerate(orders, 1):
            model.train()
            model.encoder.model.eval()
            for start in range(0, len(order), accumulation):
                batch = order[start : start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                for i in batch:
                    use_auxiliary = arm["auxiliary_weight"] > 0 and bool(eligible[i])
                    with supervisor.observe(use_auxiliary):
                        logits, hidden = forward(model, records[rows[i]["id"]], data.parent)
                        valid = tensors["valid"][i : i + 1].to(logits.device)
                        types = tensors["types"][i : i + 1].to(logits.device)
                        scale = torch.tensor(temperatures, device=logits.device)[types, None]
                        scaled = logits / scale
                        loss = distribution_losses(
                            scaled,
                            tensors["targets"][i : i + 1].to(logits.device),
                            valid,
                            types,
                            config["method"],
                        ).mean()
                        if rows[i]["retention"]:
                            reference = (
                                tensors["parent_logits"][i : i + 1].to(logits.device) / scale
                            )
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
                        if use_auxiliary:
                            if not supervisor.captured.requires_grad:
                                raise ValueError("primitive loss is detached from the backbone")
                            aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                supervisor.logits(), targets[i : i + 1].to(logits.device)
                            )
                            if auxiliary_count == 0:
                                gradients = torch.autograd.grad(
                                    aux_loss, adapters, retain_graph=True, allow_unused=True
                                )
                                norms = [
                                    float(g.detach().abs().max())
                                    for g in gradients
                                    if g is not None
                                ]
                                if (
                                    not norms
                                    or not all(math.isfinite(n) for n in norms)
                                    or max(norms) == 0
                                ):
                                    raise ValueError(
                                        "no finite nonzero primitive-to-backbone gradient"
                                    )
                                write(
                                    args.output / "auxiliary-gradient-path.json",
                                    {
                                        "backbone_parameters_with_gradient": len(norms),
                                        "backbone_parameters_with_nonzero_gradient": sum(
                                            n > 0 for n in norms
                                        ),
                                        "maximum_absolute_gradient": max(norms),
                                        "same_decision_forward": True,
                                        "optimizer_steps_before_check": steps,
                                    },
                                )
                                del gradients, norms
                            loss = loss + arm["auxiliary_weight"] * aux_loss
                            auxiliary_count += 1
                            auxiliary_sum += float(aux_loss.detach())
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite training objective")
                        (loss / len(batch)).backward()
                        total_loss += float(loss.detach())
                        processed += 1
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
                        "epoch": epoch,
                        "processed": processed,
                        "total": len(train) * len(orders),
                        "steps": steps,
                        "mean_loss": total_loss / processed,
                        "auxiliary_bce": auxiliary_sum / auxiliary_count
                        if auxiliary_count
                        else None,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    write(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            result = assess(
                model,
                supervisor,
                arm["auxiliary_weight"] > 0,
                records,
                data.parent,
                rows,
                tensors,
                dev,
                targets,
                eligible,
            )
            checkpoint = args.output / f"epoch-{epoch}" / "checkpoint"
            metadata = {
                "completed": True,
                "intermediate": True,
                "method": "workspace-" + args.arm,
                "seed": config["seed"],
                "epochs": epoch,
                "optimizer_steps": steps,
                "backbone_updated": True,
                "parent_weights_sha256": config["parent_weights_sha256"],
                "dataset_sha256": digest(data),
                "protocol_sha256": digest(args.output / "protocol.json"),
                "selected_dev": result["metrics"],
                "selection": selection_key(result["metrics"], baseline),
                "development_risk_diagnostic": result["development_risk_diagnostic"],
                "auxiliary_diagnostic": result.get("auxiliary_diagnostic"),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "final_evaluated_at_selection": False,
                "smoke": False,
            }
            model.save(checkpoint, metadata)
            if arm["auxiliary_weight"] > 0:
                save_file(
                    {
                        name: value.detach().cpu().contiguous()
                        for name, value in supervisor.decoder.state_dict().items()
                    },
                    checkpoint.parent / "auxiliary-training-only.safetensors",
                )
            history.append({"checkpoint": str(checkpoint), **metadata})
            write(args.output / "history.json", history)
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "selection": metadata["selection"],
                        "auxiliary": result.get("auxiliary_diagnostic"),
                    }
                ),
                flush=True,
            )
    finally:
        supervisor.close()
    write(
        args.output / "complete.json",
        {
            "completed": True,
            "candidates": [item["checkpoint"] for item in history],
            "release_allowed": False,
        },
    )


if __name__ == "__main__":
    main()
