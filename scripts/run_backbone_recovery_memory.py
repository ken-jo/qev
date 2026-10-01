"""One declared late-layer replay pass, followed by merged development selection."""

import gc
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from study_readout_recovery import cache, decide, read
from train_foundation_head import write
from train_workflow_backbone import forward
from train_workflow_workspace import development_risk
from workflow_depth_common import layer_number, training_mode

from veyra.data import read_records
from veyra.option_model import OptionModel
from veyra.transfer_learning import transfer_metrics
from veyra.workflow_learning import summarize_logits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(logits, tensors, rows, temperatures):
    workflow = torch.tensor([i for i, row in enumerate(rows) if row["corpus"] == "workflow"])
    legacy = {}
    for modality in ("text", "image"):
        indices = torch.tensor(
            [
                i
                for i, row in enumerate(rows)
                if row["corpus"] == "legacy" and row["modality"] == modality
            ]
        )
        legacy[modality] = transfer_metrics(
            logits[indices],
            tensors["targets"][indices],
            tensors["valid"][indices],
            tensors["gold"][indices],
            tensors["types"][indices],
        )
    return {
        "workflow": summarize_logits(logits[workflow], tensors, rows, workflow, temperatures),
        "legacy": legacy,
        "development_risk": development_risk(logits[workflow], tensors, workflow, temperatures),
    }


def main():
    config_path = Path("configs/workflow-backbone-recovery-v13.json")
    config = read(config_path)
    root = Path("runs/backbone-recovery-memory-v13")
    repair_path = Path("configs/workflow-backbone-memory-repair-v13.json")
    repair = read(repair_path)
    if digest(Path(__file__)) != repair["retry_source_sha256"]:
        raise ValueError("Memory repair source changed")
    for path, sha in repair["preserved_evidence"].items():
        if digest(path) != sha:
            raise ValueError("Preserved failure evidence changed: " + path)
    if root.exists():
        raise FileExistsError("Backbone recovery output is immutable")
    parent = Path(config["parent"])
    parent_manifest = read(parent / "manifest.json")
    parent_state = load_file(parent / "head.safetensors")
    if (
        digest(parent / "manifest.json") != config["parent_manifest_sha256"]
        or digest(parent / "head.safetensors") != config["parent_weights_sha256"]
        or digest(Path(config["features"]) / "manifest.json") != config["features_manifest_sha256"]
    ):
        raise ValueError("Declared parent or features changed")
    tensors, rows = cache("train")
    development, dev_rows = cache("dev")
    if len(rows) != 9129 or len(dev_rows) != 5102:
        raise ValueError("Declared training/development size changed")
    records, roots = {}, {}
    for corpus, spec in config["source_datasets"].items():
        path = Path(spec["path"])
        if digest(path) != spec["sha256"]:
            raise ValueError("Source dataset changed")
        for record in read_records(path):
            if record.split in {"train", "dev"}:
                records[(corpus, record.id)] = record
        roots[corpus] = path.parent
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        record = records[(row["corpus"], row["id"])]
        if record.split != "train" or record.group_id != row["group"]:
            raise ValueError("Training row identity mismatch")
        groups[(row["corpus"], row["group"])].append(i)
    units = list(groups.values())
    random.Random(config["seed"]).shuffle(units)
    boundaries = [int(len(units) * f) for f in config["checkpoint_group_fractions"]]
    segments, previous = [], 0
    for boundary in boundaries:
        segments.append([i for unit in units[previous:boundary] for i in unit])
        previous = boundary
    if sorted(i for segment in segments for i in segment) != list(range(len(rows))):
        raise ValueError("Training schedule omits or repeats rows")
    parent_metrics = read("runs/readout-recovery-v13/interpolation-protocol.json")[
        "parent_development"
    ]
    baseline = read("runs/workflow-v12-workspace-selection/merged-baseline-dev.json")["metrics"]
    source_paths = [
        Path(__file__),
        Path("scripts/study_readout_recovery.py"),
        Path("scripts/train_workflow_backbone.py"),
        Path("scripts/train_workflow_workspace.py"),
        Path("scripts/workflow_depth_common.py"),
        *Path("src/veyra").glob("*.py"),
    ]
    protocol = {
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_sha256": digest(config_path),
        "memory_repair_sha256": digest(repair_path),
        "parent_weights_sha256": config["parent_weights_sha256"],
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "source_files": {str(p): digest(p) for p in source_paths},
        "training_order_sha256": hashlib.sha256(json.dumps(segments).encode()).hexdigest(),
        "training_questions": len(rows),
        "training_groups": len(groups),
        "segment_questions": list(map(len, segments)),
        "development_questions": len(dev_rows),
        "calibration_or_final_used": False,
        "release_allowed": False,
    }
    write(root / "protocol.json", protocol)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.90)
    model = OptionModel.load(parent, local_files_only=True, merge=False)
    if len(model.encoder.model.language_model.layers) != 24:
        raise ValueError("Unexpected language depth")
    model.requires_grad_(False)
    adapters = []
    trained_keys = set()
    for name, parameter in model.named_parameters():
        if ".lora_" in name and layer_number(name) in config["trainable_adapter_layers"]:
            parameter.requires_grad_(True)
            adapters.append(parameter)
            trained_keys.add(name)
    model.readout.requires_grad_(True)
    model.binding_head.requires_grad_(True)
    model.encoder.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for index, layer in enumerate(model.encoder.model.language_model.layers):
        layer.gradient_checkpointing = index in config["trainable_adapter_layers"]
    head = list(model.readout.parameters()) + list(model.binding_head.parameters())
    trainable = adapters + head
    optimizer = torch.optim.AdamW(
        [
            {"params": adapters, "lr": config["adapter_learning_rate"]},
            {"params": head, "lr": config["readout_learning_rate"]},
        ],
        weight_decay=config["weight_decay"],
    )
    total_steps = sum(math.ceil(len(segment) / config["accumulation"]) for segment in segments)

    def rate(step):
        if step < 30:
            return (step + 1) / 30
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1, (step - 30) / max(1, total_steps - 30))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    families = {
        corpus: {row["family"] for row in rows if row["corpus"] == corpus}
        for corpus in ("workflow", "legacy")
    }
    counts = Counter((row["corpus"], row["family"]) for row in rows)
    weights = [
        0.5 * len(rows) / (len(families[row["corpus"]]) * counts[(row["corpus"], row["family"])])
        for row in rows
    ]
    temperatures = parent_manifest["calibration"]["temperatures"]
    scales = torch.tensor(temperatures, device=model.encoder.device)
    trained_keys.update(
        key for key in parent_state if key == "readout.weight" or key.startswith("binding_head.")
    )
    frozen_keys = set(parent_state) - trained_keys
    write(
        root / "parameter-plan.json",
        {
            "trainable_names": sorted(
                name for name, p in model.named_parameters() if p.requires_grad
            ),
            "trainable_parameters": sum(p.numel() for p in trainable),
            "frozen_stored_names": sorted(frozen_keys),
            "optimizer_steps": total_steps,
            "checkpointed_layers": config["trainable_adapter_layers"],
        },
    )

    def train_question(model, i, batch_length):
        row = rows[i]
        logits, _ = forward(model, records[(row["corpus"], row["id"])], roots[row["corpus"]])
        valid = tensors["valid"][i : i + 1].to(logits.device)
        target = tensors["targets"][i : i + 1].to(logits.device)
        logp = (logits / scales[tensors["types"][i]]).masked_fill(~valid, -1e9).log_softmax(-1)
        loss = -(target * logp).sum()
        if row["corpus"] == "workflow":
            teacher = tensors["parent_logits"][i : i + 1].to(logits.device) / 2
            teacher = teacher.masked_fill(~valid, -1e9).softmax(-1)
            student = (logits / 2).masked_fill(~valid, -1e9).log_softmax(-1)
            loss = loss + 4 * (teacher * (teacher.clamp_min(1e-12).log() - student)).sum()
        loss = loss * weights[i]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        (loss / batch_length).backward()
        return float(loss.detach())

    started, steps, processed, loss_sum = time.perf_counter(), 0, 0, 0.0
    candidates = []
    torch.cuda.reset_peak_memory_stats()
    for fraction, segment in zip(config["checkpoint_group_fractions"], segments, strict=True):
        training_mode(model)
        for offset in range(0, len(segment), config["accumulation"]):
            batch = segment[offset : offset + config["accumulation"]]
            optimizer.zero_grad(set_to_none=True)
            for i in batch:
                try:
                    loss_sum += train_question(model, i, len(batch))
                except Exception as error:
                    write(
                        root / "failed-training-row.json",
                        {
                            "index": i,
                            "corpus": rows[i]["corpus"],
                            "id": rows[i]["id"],
                            "input_tokens": rows[i]["input_tokens"],
                            "processed": processed,
                            "steps": steps,
                            "error": f"{type(error).__name__}: {error}",
                            "release_allowed": False,
                        },
                    )
                    raise
                processed += 1
            torch.nn.utils.clip_grad_norm_(
                trainable, config["clip_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            steps += 1
            gc.collect()
            torch.cuda.empty_cache()
            if steps % 25 == 0:
                progress = {
                    "stage": "training",
                    "processed": processed,
                    "total": len(rows),
                    "steps": steps,
                    "mean_loss": loss_sum / processed,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "release_allowed": False,
                }
                write(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        state = model.trainable_state()
        if any(not torch.equal(state[key], parent_state[key]) for key in frozen_keys):
            raise ValueError("Frozen stored parameter changed")
        path = root / f"fraction-{fraction:g}" / "checkpoint"
        model.save(
            path,
            {
                "completed": True,
                "intermediate": False,
                "optimizer_steps": parent_manifest["training"]["optimizer_steps"] + steps,
                "additional_optimizer_steps": steps,
                "additional_questions": processed,
                "backbone_recovery_design_sha256": digest(config_path),
                "backbone_recovery_protocol_sha256": digest(root / "protocol.json"),
                "parent_weights_sha256": config["parent_weights_sha256"],
                "parent_manifest_sha256": config["parent_manifest_sha256"],
                "fraction": fraction,
                "seed": config["seed"],
                "trained_layers": config["trainable_adapter_layers"],
                "frozen_stored_parameters_exact": True,
                "release_allowed": False,
                "calibration_status": "Inherited parent policy; development only until refit",
            },
        )
        optimizer_path = path.parent / "optimizer-state.pt"
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "steps": steps,
                "processed": processed,
                "loss_sum": loss_sum,
            },
            optimizer_path,
        )
        write(
            path.parent / "optimizer-state.json",
            {
                "sha256": digest(optimizer_path),
                "weights_sha256": digest(path / "head.safetensors"),
                "steps": steps,
                "processed": processed,
                "private_local_resumption_only": True,
            },
        )
        candidates.append(str(path))
        write(
            root / "training-checkpoints.json",
            {
                "candidates": candidates,
                "steps": steps,
                "processed": processed,
                "release_allowed": False,
            },
        )
    write(
        root / "training-complete.json",
        {
            "completed": True,
            "candidates": candidates,
            "questions": processed,
            "steps": steps,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "source_files": protocol["source_files"],
            "release_allowed": False,
        },
    )
    del optimizer, scheduler, model, adapters, head, trainable, layer, parameter, train_question
    gc.collect()
    torch.cuda.empty_cache()
    results = []
    for candidate in candidates:
        model = OptionModel.load(candidate, local_files_only=True, merge=True)
        collected = []
        with torch.inference_mode():
            for i, row in enumerate(dev_rows):
                values, _ = forward(
                    model, records[(row["corpus"], row["id"])], roots[row["corpus"]]
                )
                collected.append(values[0].cpu())
                if (i + 1) % 200 == 0:
                    write(
                        root / "progress.json",
                        {
                            "stage": "merged-development",
                            "candidate": candidate,
                            "evaluated": i + 1,
                            "total": len(dev_rows),
                            "release_allowed": False,
                        },
                    )
        logits = torch.stack(collected)
        result_metrics = metrics(logits, development, dev_rows, temperatures)
        candidate_state = load_file(Path(candidate) / "head.safetensors")
        movement = sum(
            float((candidate_state[k] - parent_state[k]).square().sum()) for k in candidate_state
        )
        checks, rank = decide(result_metrics, parent_metrics, baseline, config, movement)
        folder = Path(candidate).parent
        save_file({"logits": logits}, folder / "merged-development.safetensors")
        result = {
            "checkpoint": candidate,
            "weights_sha256": digest(Path(candidate) / "head.safetensors"),
            "manifest_sha256": digest(Path(candidate) / "manifest.json"),
            "metrics": result_metrics,
            "checks": checks,
            "selection": rank,
            "merged_development_logits_sha256": digest(folder / "merged-development.safetensors"),
            "merged_bf16_deployment": True,
            "final_or_calibration_used": False,
        }
        write(folder / "development.json", result)
        results.append(result)
        print(json.dumps({"candidate": candidate, "checks": checks, "rank": rank}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    selected = max(results, key=lambda row: tuple(row["selection"]))
    if any(digest(p) != sha for p, sha in protocol["source_files"].items()):
        raise ValueError("Study source changed during execution")
    write(
        root / "selection.json",
        {
            "selected": selected,
            "eligible": bool(selected["selection"][0]),
            "candidates": results,
            "design_sha256": digest(config_path),
            "memory_repair_sha256": digest(repair_path),
            "release_protocol_sha256": config["release_protocol_sha256"],
            "final_or_calibration_used": False,
            "release_allowed": False,
        },
    )
    write(
        root / "progress.json",
        {"stage": "complete", "eligible": bool(selected["selection"][0]), "release_allowed": False},
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        output = Path("runs/backbone-recovery-memory-v13/execution-failure.json")
        if not output.exists():
            write(
                output,
                {
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                    "release_allowed": False,
                },
            )
        raise
