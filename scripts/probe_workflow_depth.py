"""Measure earlier-layer gradient/checkpointing feasibility with zero parameter updates."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from train_foundation_head import write
from workflow_depth_common import expand_depth, optimizer_groups, training_mode
from workflow_skill_common import canonical_target

from veyra.data import TrainingRecord
from veyra.option_model import OptionModel
from veyra.proper_learning import distribution_losses


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.trainable_state().items()):
        value.update(name.encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def forward(model, record, root):
    states, _, tokens, conditions = model.encode(record.request, root, capture_condition=True)
    logits = model.readout(states) + model.binding_head(states, model.condition_readout(conditions))
    return logits, tokens


def main(output):
    config_path = Path("configs/workflow-depth-feasibility-v12.json")
    config = read(config_path)
    paths = {
        Path(config["records"]): config["records_sha256"],
        Path(config["selection"]): config["selection_sha256"],
        Path(config["parent"]) / "head.safetensors": config["parent_weights_sha256"],
        Path(config["parent"]) / "manifest.json": config["parent_manifest_sha256"],
    }
    for path, expected in paths.items():
        if digest(path) != expected:
            raise ValueError("feasibility input changed: " + str(path))
    selected = read(config["selection"])
    if selected["eligible"] or selected["weights_sha256"] != config["parent_weights_sha256"]:
        raise ValueError("require the highest-ranked, ineligible completed-study checkpoint")
    records = {}
    for line in Path(config["records"]).open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] == "train" and raw["id"] in config["record_ids"]:
            records[raw["id"]] = TrainingRecord.model_validate(raw)
    if set(records) != set(config["record_ids"]):
        raise ValueError("feasibility cases must all be original training records")
    sources = {
        str(path): digest(path)
        for path in (
            Path(__file__),
            Path("scripts/workflow_depth_common.py"),
            Path("scripts/workflow_skill_common.py"),
            Path("src/veyra/option_model.py"),
            Path("src/veyra/proper_learning.py"),
        )
    }
    write(
        output / "protocol.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": digest(config_path),
            "source_sha256": sources,
            "learning_rate": 0.0,
            "calibration_or_final_used": False,
        },
    )
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    model = OptionModel.load(config["parent"], local_files_only=True, merge=False)
    first = records[config["record_ids"][0]]
    root = Path(config["records"]).parent
    with torch.no_grad():
        reference, _ = forward(model, first, root)
    expansion = expand_depth(model)
    with torch.no_grad():
        expanded, _ = forward(model, first, root)
    difference = float((reference - expanded).abs().max())
    if difference > 1e-5:
        raise ValueError("zero-output expansion changed initial logits: " + str(difference))
    groups = optimizer_groups(model, 0.0, 0.0, 0.0)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    initial_hash = state_hash(model)
    results = []
    for identifier in config["record_ids"]:
        record = records[identifier]
        optimizer.zero_grad(set_to_none=True)
        training_mode(model)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits, tokens = forward(model, record, root)
        target, valid, kind = canonical_target(record)
        target, valid = target[None].to(logits.device), valid[None].to(logits.device)
        types = torch.tensor([kind], device=logits.device)
        scale = model.calibration.temperatures[kind]
        loss = distribution_losses(logits / scale, target, valid, types, "rloo").mean()
        loss.backward()
        gradient_norms = {}
        for group in groups:
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in group["params"]):
                raise ValueError("missing or nonfinite gradient in " + group["name"])
            gradient_norms[group["name"]] = float(
                torch.stack([p.grad.float().square().sum() for p in group["params"]]).sum().sqrt()
            )
            if gradient_norms[group["name"]] <= 0:
                raise ValueError("zero gradient for an entire parameter group")
        torch.nn.utils.clip_grad_norm_(
            [p for group in groups for p in group["params"]], 1.0, error_if_nonfinite=True
        )
        # Zero rates allocate real AdamW state while leaving every parameter value unchanged.
        optimizer.step()
        torch.cuda.synchronize()
        if state_hash(model) != initial_hash:
            raise ValueError("zero-learning-rate feasibility step changed model parameters")
        result = {
            "record_id": identifier,
            "has_image": bool(record.request.state.images),
            "input_tokens": tokens,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "seconds": time.perf_counter() - started,
            "gradient_norms": gradient_norms,
        }
        results.append(result)
        write(output / "progress.json", {"status": "running", "cases": results})
        print(json.dumps(result), flush=True)
        del logits, loss
    for path, expected in paths.items():
        if digest(path) != expected:
            raise ValueError("original input changed during feasibility measurement")
    write(
        output / "results.json",
        {
            "completed": True,
            "expansion": expansion,
            "initial_logits_maximum_difference": difference,
            "parameter_count_by_group": {
                g["name"]: sum(p.numel() for p in g["params"]) for g in groups
            },
            "cases": results,
            "parameter_values_unchanged": True,
            "configuration_sha256": digest(config_path),
            "source_sha256": sources,
            "release_allowed": False,
            "limitation": (
                "A few declared training inputs and zero-rate AdamW steps only. "
                "This measures memory and gradients, not learning, development performance, "
                "all possible input shapes, runtime latency or release acceptance."
            ),
        },
    )
    write(output / "progress.json", {"status": "complete", "release_allowed": False})


if __name__ == "__main__":
    directory = Path("runs/workflow-v12-depth-feasibility")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        main(directory)
    except Exception as error:
        write(directory / "progress.json", {"status": "failed", "error": str(error)})
        raise
