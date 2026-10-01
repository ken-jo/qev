"""Supervised readout + late-layer LoRA; select using development only."""

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from veyra.candidates import candidates_for
from veyra.constants import MODEL_ID, MODEL_REVISION
from veyra.data import read_records
from veyra.interventions import augment_unit, retype_unit, training_units
from veyra.option_model import OptionModel
from veyra.replay import audit_replay, replay_pool, sample_replay


@torch.inference_mode()
def evaluate(model, records, root):
    model.eval()
    rows = []
    for record in records:
        logits, _ = model(record.request, root)
        for name, value in logits.items():
            candidates = candidates_for(record.request.questions[name])
            target = torch.tensor(
                [record.targets[name][c.key] for c in candidates], device=value.device
            )
            winner = int(value.argmax())
            rows.append(
                {
                    "family": record.family,
                    "type": record.request.questions[name].type,
                    "correct": float(target[winner]),
                    "nll": float(-(target * value.log_softmax(-1)).sum()),
                }
            )
    result = {
        "questions": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "nll": sum(r["nll"] for r in rows) / len(rows),
    }
    for modality in ("text", "image"):
        subset = [r for r in rows if r["family"].startswith(modality + "_")]
        if subset:
            result[modality + "_accuracy"] = sum(r["correct"] for r in subset) / len(subset)
    for field in ("family", "type"):
        result["by_" + field] = {}
        for value in sorted({r[field] for r in rows}):
            subset = [r for r in rows if r[field] == value]
            result["by_" + field][value] = sum(r["correct"] for r in subset) / len(subset)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("data/policy-v3/records.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument(
        "--initialize", type=Path, help="Initialize compatible adapters; optimizer starts fresh"
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--selection", choices=["nll", "min-modality-accuracy"], default="nll")
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--interventions", action="store_true")
    parser.add_argument("--type-interventions", action="store_true")
    parser.add_argument("--reasoning-slots", type=int, default=0)
    parser.add_argument("--condition-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--replay-records", type=Path)
    parser.add_argument("--replay-per-epoch", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("training output must be empty")
    if args.epochs < 1 or args.accumulation < 1 or args.learning_rate <= 0:
        raise ValueError("training settings must be positive")
    if args.replay_per_epoch < 0 or bool(args.replay_records) != bool(args.replay_per_epoch):
        raise ValueError("replay requires both a dataset and a positive request count")
    if args.type_interventions and not args.interventions:
        raise ValueError("type interventions require the controlled intervention training path")
    if args.reasoning_slots and (not args.type_interventions or args.condition_weight <= 0):
        raise ValueError("internal condition supervision requires typed training annotations")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            Path("src/veyra/option_model.py"),
            Path("src/veyra/interventions.py"),
            Path("src/veyra/evidence_data.py"),
            Path("src/veyra/policy_data.py"),
            Path("src/veyra/replay.py"),
            Path("src/veyra/reasoning_workspace.py"),
        )
    }
    (args.output / "run.json").write_text(
        json.dumps({"arguments": vars(args), "source_sha256": source_hashes}, default=str, indent=2)
        + "\n",
        encoding="utf-8",
    )
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    records = read_records(args.records)
    train = [r for r in records if r.split == "train"]
    dev = [r for r in records if r.split == "dev"]
    replay = {}
    replay_audit = None
    if args.replay_records:
        replay_train = [r for r in read_records(args.replay_records) if r.split == "train"]
        replay_audit = audit_replay(
            records, args.records.parent, replay_train, args.replay_records.parent
        )
        replay = replay_pool(replay_train)
    if args.smoke:
        train = train[:8] + [r for r in train if r.family.startswith("image_")][:8]
        dev = dev[:4] + [r for r in dev if r.family.startswith("image_")][:4]
    units = training_units(train) if args.interventions else [[record] for record in train]
    epoch_records = sum(len(unit) for unit in units) + args.replay_per_epoch
    model = OptionModel.create(
        layers=args.layers, rank=args.rank, reasoning_slots=args.reasoning_slots
    )
    parent = None
    if args.initialize:
        parent = json.loads((args.initialize / "manifest.json").read_text())
        if parent["backbone"] != {"model_id": MODEL_ID, "revision": MODEL_REVISION}:
            raise ValueError("initialization backbone revision mismatch")
        weights = args.initialize / "head.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != parent["weights_sha256"]:
            raise ValueError("initialization checkpoint checksum mismatch")
        if (
            parent["architecture"] != "option_readout"
            or parent["encoder"] != model.encoder.config.to_dict()
        ):
            raise ValueError("incompatible initialization encoding")
        if (
            parent["adaptation"]["rank"] != args.rank
            or parent["adaptation"]["alpha"] != model.adaptation["alpha"]
        ):
            raise ValueError("initialization rank and alpha must match")
        state = load_file(weights)
        if not set(state).issubset(model.trainable_state()):
            raise ValueError("initialization needs the same or more adapted layers")
        model.load_state_dict(state, strict=False)
    adapters = [p for name, p in model.named_parameters() if ".lora_" in name]
    parameter_groups = [
        {"params": adapters, "lr": args.learning_rate},
        {"params": model.readout.parameters(), "lr": args.learning_rate * 0.1},
    ]
    if args.reasoning_slots:
        parameter_groups.append(
            {"params": model.condition_readout.parameters(), "lr": args.learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)
    total_steps = math.ceil(epoch_records / args.accumulation) * args.epochs

    def rate(step):
        if args.schedule == "constant":
            return 1.0
        warmup = min(max(args.warmup_steps, 0), total_steps - 1)
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, dev, args.records.parent)
    print(json.dumps({"initial_dev": initial}), flush=True)
    best, best_value, history, steps = None, None, [], 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        shuffled = [(unit, args.records.parent, args.interventions) for unit in units]
        if replay:
            shuffled.extend(
                ([record], args.replay_records.parent, False)
                for record in sample_replay(replay, args.replay_per_epoch, rng)
            )
        rng.shuffle(shuffled)
        order = []
        for unit, root, controlled in shuffled:
            if controlled and args.type_interventions:
                unit = retype_unit(unit, rng)
            chosen = augment_unit(unit, rng) if controlled else list(unit)
            rng.shuffle(chosen)
            order.extend((record, root, controlled) for record in chosen)
        model.train()
        model.encoder.model.eval()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        previous_group, previous_correct_probability = None, None
        for i, (record, root, controlled) in enumerate(order):
            branch_tags = [tag for tag in record.tags if tag.startswith("oracle_condition_branch:")]
            supervised_condition = (
                args.reasoning_slots
                and controlled
                and "training_auxiliary_condition" not in record.tags
            )
            if supervised_condition:
                if len(branch_tags) != 1:
                    raise ValueError(
                        "controlled workspace training needs one audited branch target"
                    )
                logits, _, conditions = model.forward_with_condition(record.request, root)
            else:
                logits, _ = model(record.request, root)
            losses = []
            for name, value in logits.items():
                candidates = candidates_for(record.request.questions[name])
                target = torch.tensor(
                    [record.targets[name][c.key] for c in candidates], device=value.device
                )
                losses.append(-(target * value.log_softmax(-1)).sum())
            loss = torch.stack(losses).mean()
            if supervised_condition:
                branch = int(branch_tags[0].split(":")[1])
                condition_loss = torch.stack(
                    [-value.log_softmax(-1)[branch] for value in conditions.values()]
                ).mean()
                loss = loss + args.condition_weight * condition_loss
            if controlled:
                # The two rules have different target meanings. Match their probability
                # of satisfying the rule, while CE still anchors both to ground truth.
                correct_probability = value.softmax(-1)[target.argmax()]
                if previous_group == record.group_id:
                    first = torch.stack(
                        (previous_correct_probability, 1 - previous_correct_probability)
                    )
                    second = torch.stack((correct_probability, 1 - correct_probability))
                    midpoint = (first + second) / 2
                    consistency = 0.5 * (
                        (
                            first * (first.clamp_min(1e-7).log() - midpoint.clamp_min(1e-7).log())
                        ).sum()
                        + (
                            second * (second.clamp_min(1e-7).log() - midpoint.clamp_min(1e-7).log())
                        ).sum()
                    )
                    loss = loss + args.consistency_weight * consistency
                previous_group = record.group_id
                previous_correct_probability = correct_probability.detach()
            else:
                previous_group, previous_correct_probability = None, None
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            total += float(loss.detach())
            window = min(
                args.accumulation, len(order) - (i // args.accumulation) * args.accumulation
            )
            (loss / window).backward()
            if (i + 1) % args.accumulation == 0 or i == len(order) - 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1
                )
                if not torch.isfinite(grad_norm):
                    raise ValueError("nonfinite training gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            if (i + 1) % 100 == 0 or i == len(order) - 1:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "records": i + 1,
                            "total": len(order),
                            "loss": total / (i + 1),
                            "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                            "adapter_lr": optimizer.param_groups[0]["lr"],
                        }
                    ),
                    flush=True,
                )
        metrics = evaluate(model, dev, args.records.parent)
        history.append({"epoch": epoch, "train_loss": total / len(order), "dev": metrics})
        print(json.dumps(history[-1]), flush=True)
        selection_value = (
            (min(metrics["text_accuracy"], metrics["image_accuracy"]), -metrics["nll"])
            if args.selection == "min-modality-accuracy"
            else (-metrics["nll"],)
        )
        if best_value is None or selection_value > best_value:
            best, best_value = copy.deepcopy(model.trainable_state()), selection_value
            selected_epoch = epoch
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
        model.save(
            args.output / "epochs" / f"epoch-{epoch}",
            {
                "completed": True,
                "optimizer_steps": steps,
                "epochs": epoch,
                "selection_split": "dev",
                "history": history,
                "seed": args.seed,
                "initialization_sha256": parent["weights_sha256"] if parent else None,
                "dataset_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
                "intermediate": True,
            },
        )
    model.load_state_dict(best, strict=False)
    # Progress is separate from the immutable checkpoint subdirectory.
    metadata = {
        "completed": True,
        "optimizer_steps": steps,
        "epochs": args.epochs,
        "selected_epoch": selected_epoch,
        "selection_split": "dev",
        "selection_metric": args.selection,
        "initial_dev": initial,
        "history": history,
        "seed": args.seed,
        "initialization_sha256": parent["weights_sha256"] if parent else None,
        "optimizer_resumed": False,
        "source_sha256": source_hashes,
        "learning_rate": args.learning_rate,
        "schedule": args.schedule,
        "warmup_steps": args.warmup_steps,
        "interventions": args.interventions,
        "type_interventions": args.type_interventions,
        "reasoning_slots": args.reasoning_slots,
        "condition_weight": args.condition_weight if args.reasoning_slots else 0,
        "consistency_weight": args.consistency_weight if args.interventions else 0,
        "epoch_records_including_auxiliary": epoch_records,
        "readout_lr": args.learning_rate * 0.1,
        "accumulation": args.accumulation,
        "train_records": len(train),
        "replay_requests_per_epoch": args.replay_per_epoch,
        "replay_audit": replay_audit,
        "replay_dataset_sha256": (
            hashlib.sha256(args.replay_records.read_bytes()).hexdigest()
            if args.replay_records
            else None
        ),
        "smoke": args.smoke,
        "dataset_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "elapsed_seconds": time.perf_counter() - started,
    }
    model.save(args.output / "checkpoint", metadata)
    print(
        json.dumps({"saved": str(args.output / "checkpoint"), "selected_epoch": selected_epoch}),
        flush=True,
    )


if __name__ == "__main__":
    main()
