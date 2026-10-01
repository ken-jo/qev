"""Cross-fit readout errors, then compare a small learned confidence head on development."""

import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from train_foundation_head import write

from veyra.constants import QUESTION_TYPES
from veyra.data import read_records
from veyra.proper_learning import distribution_losses, grouped_batches, retention_losses
from veyra.transfer_learning import TransferReadout
from veyra.workflow_learning import SHARES, annotate


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_hash(seed, group):
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def expected_error(logits, tensors, indices):
    prediction = logits.masked_fill(~tensors["valid"][indices], -1e9).argmax(-1)
    gold = tensors["gold"][indices]
    error = 1 - tensors["targets"][indices].gather(1, prediction[:, None]).squeeze(1)
    return torch.where(gold >= 0, (prediction != gold).float(), error), prediction


def confidence_inputs(states, logits, valid, types, temperatures):
    count = valid.sum(-1).float()
    probabilities = (
        (logits / torch.tensor(temperatures)[types, None]).masked_fill(~valid, -1e9).softmax(-1)
    )
    top = probabilities.topk(2, dim=-1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1) / count.log()
    summary = torch.stack(
        (top[:, 0], top[:, 0] - top[:, 1], entropy, probabilities.square().sum(-1), count / 16),
        dim=-1,
    )
    return torch.cat(
        (F.layer_norm(states, (states.shape[1],)), summary, F.one_hot(types, 3).float()), dim=-1
    )


def rank_report(confidence, logits, tensors, indices, rows):
    error, prediction = expected_error(logits, tensors, indices)
    types = tensors["types"][indices]

    def summarize(selected):
        order = selected[torch.argsort(confidence[selected], descending=True, stable=True)]
        count = len(order)
        size = torch.arange(1, count + 1)
        cumulative = error[order].cumsum(0) / size
        boundary = torch.ones(count, dtype=torch.bool)
        boundary[:-1] = confidence[order[:-1]] != confidence[order[1:]]
        eligible = torch.where((cumulative <= 0.15) & (size >= 20) & boundary)[0]
        minimum = math.ceil(count * 0.6)
        return {
            "questions": count,
            "maximum_threshold_coverage_at_15pct_error": (int(eligible[-1]) + 1) / count
            if len(eligible)
            else 0.0,
            "fixed_count_error_at_60pct": float(cumulative[minimum - 1]),
            "correctness_brier": float(
                (confidence[selected] - (1 - error[selected])).square().mean()
            ),
        }

    critical = torch.tensor([rows[i]["critical_position"] for i in indices.tolist()])
    mass = tensors["targets"][indices].gather(1, critical.clamp_min(0)[:, None]).squeeze(1)
    cost = error + 4 * mass * (critical >= 0) * (prediction != critical)
    uncertain = torch.tensor(
        [
            i
            for i, original in enumerate(indices.tolist())
            if rows[original]["domain"] == "uncertainty"
        ]
    )
    order = uncertain[torch.argsort(confidence[uncertain], descending=True, stable=True)]
    accepted = order[: math.floor(len(order) * 0.8)]
    return {
        "by_type": {
            kind: summarize(torch.where(types == type_id)[0])
            for type_id, kind in enumerate(QUESTION_TYPES)
        },
        "uncertainty_cost_at_80pct": float(cost[accepted].mean()),
        "uncertainty_answer_count_at_80pct": len(accepted),
        "class_predictions_and_distributions_unchanged": True,
    }


def main():
    config_path = Path("configs/workflow-reliability-pilot-v12.json")
    config = read(config_path)
    output = Path("runs/workflow-v12-reliability-pilot")
    if output.exists():
        raise FileExistsError("reliability pilot is immutable")
    features, parent_path, classifier_path = map(
        Path, (config["features"], config["representation_parent"], config["classifier"])
    )
    cache, specification = read(features / "manifest.json"), read(features / "specification.json")
    if (
        config["status"] != "declared_before_training"
        or digest(config["records"]) != config["records_sha256"]
        or config["records_sha256"] != specification["records_sha256"]
        or cache["complete"] is not True
        or specification["calibration_or_final_encoded"] is not False
        or digest(classifier_path / "head.safetensors") != config["classifier_weights_sha256"]
        or digest(classifier_path / "manifest.json") != config["classifier_manifest_sha256"]
        or digest(parent_path / "head.safetensors") != specification["parent_weights_sha256"]
        or digest(parent_path / "manifest.json") != specification["parent_manifest_sha256"]
        or digest("runs/workflow-v12-head/selection.json") != config["classifier_selection_sha256"]
    ):
        raise ValueError("declared data, parent or classifier changed")
    for filename, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(features / filename) != cache[key]:
            raise ValueError("feature cache changed")
    tensors, rows = load_file(features / "features.safetensors"), read(features / "records.json")
    records = [r for r in read_records(Path(config["records"])) if r.split in {"train", "dev"}]
    annotate(rows, records)
    train = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "train"])
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    if {rows[i]["group"] for i in train.tolist()} & {rows[i]["group"] for i in dev.tolist()}:
        raise ValueError("train/development groups overlap")
    families = sorted(
        {rows[i]["family"] for i in train.tolist() if rows[i]["domain"] == "workflow_new"},
        key=lambda key: group_hash(config["fold_seed"], key),
    )
    if len(families) != 6:
        raise ValueError("expected the six original procedural training families")
    family_folds = {family: i % config["folds"] for i, family in enumerate(families)}
    assignments = {
        rows[i]["group"]: (
            family_folds[rows[i]["family"]]
            if rows[i]["domain"] in {"workflow_new", "uncertainty"}
            else int(group_hash(config["fold_seed"], rows[i]["group"]), 16) % config["folds"]
        )
        for i in train.tolist()
    }
    protocol = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(config_path),
        "configuration": config,
        "source_sha256": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("src/veyra/proper_learning.py"),
                Path("src/veyra/transfer_learning.py"),
                Path("src/veyra/workflow_learning.py"),
            )
        },
        "feature_cache": cache,
        "family_folds": family_folds,
        "group_fold_assignments": assignments,
        "calibration_or_final_used": False,
        "active_candidate_lists_changed": False,
    }
    write(output / "protocol.json", protocol)
    torch.set_num_threads(4)
    parent, parent_state = (
        read(parent_path / "manifest.json"),
        load_file(parent_path / "head.safetensors"),
    )
    classifier_state = load_file(classifier_path / "head.safetensors")
    if set(classifier_state) != set(parent_state) or any(
        not torch.equal(value, parent_state[key])
        for key, value in classifier_state.items()
        if key != "readout.weight" and not key.startswith("binding_head.")
    ):
        raise ValueError("classifier does not use the frozen representation parent")
    temperatures = parent["calibration"]["temperatures"]
    scale = torch.tensor(temperatures)[tensors["types"], None]
    replay = torch.tensor([row["retention"] for row in rows])
    oof = torch.zeros(len(rows), 16)
    observations = torch.zeros(len(rows), dtype=torch.long)
    for fold in range(config["folds"]):
        fit = torch.tensor([i for i in train.tolist() if assignments[rows[i]["group"]] != fold])
        hold = torch.tensor([i for i in train.tolist() if assignments[rows[i]["group"]] == fold])
        held_families = {family for family, value in family_folds.items() if value == fold}
        if len(held_families) != 2 or any(
            rows[i]["family"] in held_families
            for i in fit.tolist()
            if rows[i]["domain"] in {"workflow_new", "uncertainty"}
        ):
            raise ValueError("procedural family crossed fold boundary")
        counts = Counter(rows[i]["domain"] for i in fit.tolist())
        weights = torch.tensor(
            [
                SHARES.get(row["domain"], 0) * len(fit) / max(1, counts[row["domain"]])
                for row in rows
            ]
        )
        seed = config["fold_seed"] + fold
        torch.manual_seed(seed)
        model = TransferReadout(parent, parent_state)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.readout.parameters(), "lr": 0.0001, "weight_decay": 0.0},
                {"params": model.binding_head.parameters(), "lr": 0.001, "weight_decay": 0.01},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config["fold_epochs"], eta_min=0.00001
        )
        for epoch in range(1, config["fold_epochs"] + 1):
            model.train()
            for batch in grouped_batches(
                fit, rows, 128, torch.Generator().manual_seed(seed * 1000 + epoch)
            ):
                logits = (
                    model(tensors["states"][batch], tensors["condition_logits"][batch])
                    / scale[batch]
                )
                valid = tensors["valid"][batch]
                loss = distribution_losses(
                    logits, tensors["targets"][batch], valid, tensors["types"][batch], "rloo"
                )
                loss = loss * torch.where(replay[batch], 0.25, 1.0)
                loss += (
                    4
                    * retention_losses(
                        logits, tensors["parent_logits"][batch] / scale[batch], valid
                    )
                    * replay[batch]
                )
                objective = (loss * weights[batch]).mean()
                if not torch.isfinite(objective):
                    raise ValueError("nonfinite cross-fit objective")
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            scheduler.step()
        model.eval()
        with torch.inference_mode():
            oof[hold] = model(tensors["states"][hold], tensors["condition_logits"][hold])
        observations[hold] += 1
        save_file(
            {key: value.contiguous() for key, value in model.state_dict().items()},
            output / f"fold-{fold}.safetensors",
        )
        write(
            output / "progress.json",
            {"stage": "cross_fit", "fold_completed": fold + 1, "folds": config["folds"]},
        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "fit": len(fit),
                    "held_out": len(hold),
                    "withheld_families": sorted(held_families),
                }
            ),
            flush=True,
        )
    if not (observations[train] == 1).all() or (observations[dev] != 0).any():
        raise ValueError("out-of-fold coverage is not exact")
    save_file({"logits": oof[train], "cache_indices": train}, output / "out-of-fold.safetensors")
    xtrain = confidence_inputs(
        tensors["states"][train],
        oof[train],
        tensors["valid"][train],
        tensors["types"][train],
        temperatures,
    )
    target = 1 - expected_error(oof[train], tensors, train)[0]
    classifier = TransferReadout(read(classifier_path / "manifest.json"), classifier_state).eval()
    with torch.inference_mode():
        dev_logits = classifier(tensors["states"][dev], tensors["condition_logits"][dev])
    xdev = confidence_inputs(
        tensors["states"][dev],
        dev_logits,
        tensors["valid"][dev],
        tensors["types"][dev],
        temperatures,
    )
    baseline_confidence = (
        (dev_logits / scale[dev])
        .masked_fill(~tensors["valid"][dev], -1e9)
        .softmax(-1)
        .max(-1)
        .values
    )
    baseline = rank_report(baseline_confidence, dev_logits, tensors, dev, rows)
    counts = Counter(rows[i]["domain"] for i in train.tolist())
    weights = torch.tensor(
        [SHARES[rows[i]["domain"]] * len(train) / counts[rows[i]["domain"]] for i in train.tolist()]
    )
    results = []
    for seed in config["critic_seeds"]:
        torch.manual_seed(seed)
        critic = nn.Sequential(
            nn.Linear(xtrain.shape[1], config["critic_width"]),
            nn.GELU(),
            nn.Dropout(config["critic_dropout"]),
            nn.Linear(config["critic_width"], 1),
        )
        optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=config["critic_learning_rate"],
            weight_decay=config["critic_weight_decay"],
        )
        history = []
        for epoch in range(config["critic_epochs"]):
            critic.train()
            order = torch.randperm(
                len(train), generator=torch.Generator().manual_seed(seed * 1000 + epoch)
            )
            total = 0.0
            for batch in order.split(config["critic_batch_size"]):
                loss = (
                    F.binary_cross_entropy_with_logits(
                        critic(xtrain[batch]).squeeze(-1), target[batch], reduction="none"
                    )
                    * weights[batch]
                ).mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite confidence objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            history.append({"epoch": epoch + 1, "train_bce": total / len(train)})
        critic.eval()
        with torch.inference_mode():
            confidence = critic(xdev).squeeze(-1).sigmoid()
        report = rank_report(confidence, dev_logits, tensors, dev, rows)
        folder = output / f"critic-{seed}"
        folder.mkdir()
        save_file(
            {key: value.contiguous() for key, value in critic.state_dict().items()},
            folder / "critic.safetensors",
        )
        save_file(
            {"confidence": confidence, "classifier_logits": dev_logits, "cache_indices": dev},
            folder / "development.safetensors",
        )
        write(folder / "training.json", history)
        result = {
            "seed": seed,
            "parameters": sum(p.numel() for p in critic.parameters()),
            "development": report,
            "critic_sha256": digest(folder / "critic.safetensors"),
            "development_sha256": digest(folder / "development.safetensors"),
        }
        results.append(result)
        write(output / "results.json", results)
        write(output / "progress.json", {"stage": "critic", "seed_completed": seed})
        print(json.dumps(result), flush=True)
    median_coverage = {
        kind: statistics.median(
            row["development"]["by_type"][kind]["maximum_threshold_coverage_at_15pct_error"]
            for row in results
        )
        for kind in QUESTION_TYPES
    }
    median_cost = statistics.median(
        row["development"]["uncertainty_cost_at_80pct"] for row in results
    )
    comparison = {
        "baseline": baseline,
        "results": results,
        "median_type_coverage": median_coverage,
        "median_uncertainty_cost_at_80pct": median_cost,
        "representative_seed": config["critic_seeds"][0],
        "exploratory_screen_passed": all(value >= 0.6 for value in median_coverage.values())
        and median_cost <= baseline["uncertainty_cost_at_80pct"],
        "protocol_sha256": digest(output / "protocol.json"),
        "out_of_fold_sha256": digest(output / "out-of-fold.safetensors"),
        "calibration_or_final_used": False,
        "release_allowed": False,
        "runtime_integration_performed": False,
    }
    write(output / "comparison.json", comparison)
    write(
        output / "complete.json",
        {
            "completed": True,
            "release_allowed": False,
            "exploratory_screen_passed": comparison["exploratory_screen_passed"],
        },
    )
    print(
        json.dumps(
            {
                "complete": True,
                "screen_passed": comparison["exploratory_screen_passed"],
                "median_coverage": median_coverage,
                "median_cost": median_cost,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
