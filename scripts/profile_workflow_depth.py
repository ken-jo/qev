"""Compare zero-rate training execution and gradients on declared training inputs only."""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from probe_workflow_depth import forward, state_hash
from safetensors.torch import load_file, save_file
from train_foundation_head import write
from workflow_depth_common import expand_depth, optimizer_groups, training_mode
from workflow_depth_execution import configure_checkpointing
from workflow_skill_common import canonical_target

from veyra.data import TrainingRecord
from veyra.option_model import OptionModel
from veyra.proper_learning import distribution_losses

CONFIG = Path("configs/workflow-depth-execution-v12.json")
OUTPUT = Path("runs/workflow-v12-depth-execution")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inputs(config):
    original = read(config["feasibility_config"])
    bindings = {
        config["feasibility_config"]: config["feasibility_config_sha256"],
        original["records"]: original["records_sha256"],
        Path(original["parent"]) / "head.safetensors": original["parent_weights_sha256"],
        Path(original["parent"]) / "manifest.json": original["parent_manifest_sha256"],
    }
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError("declared profiling input changed: " + str(path))
    records = {}
    for line in Path(original["records"]).open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] == "train" and raw["id"] in original["record_ids"]:
            records[raw["id"]] = TrainingRecord.model_validate(raw)
    if set(records) != set(original["record_ids"]):
        raise ValueError("all declared profiling cases must be training inputs")
    return original, records


def child(strategy):
    config = read(CONFIG)
    original, records = inputs(config)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.cuda.set_per_process_memory_fraction(config["allocator_memory_fraction"])
    model = OptionModel.load(original["parent"], local_files_only=True, merge=False)
    expand_depth(model)
    checkpointed = configure_checkpointing(model, strategy)
    groups = optimizer_groups(model, 0.0, 0.0, 0.0)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    initial = state_hash(model)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    output = OUTPUT / strategy
    output.mkdir(exist_ok=False)
    results = []
    for index, identifier in enumerate(original["record_ids"]):
        record = records[identifier]
        row = {"record_id": identifier, "has_image": bool(record.request.state.images)}
        durations = []
        logits = loss = gradient = None
        try:
            training_mode(model)
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(config["warmups"] + config["repeats"]):
                # Replay the same score-function noise in all strategies and repetitions.
                torch.manual_seed(config["seed"] + index)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                logits, tokens = forward(model, record, Path(original["records"]).parent)
                target, valid, kind = canonical_target(record)
                target, valid = target[None].to(logits.device), valid[None].to(logits.device)
                types = torch.tensor([kind], device=logits.device)
                loss = distribution_losses(
                    logits / model.calibration.temperatures[kind], target, valid, types, "rloo"
                ).mean()
                loss.backward()
                # Save the unmodified gradient before clipping, outside the timed repetitions.
                if repeat == 0:
                    parameters = dict(model.named_parameters())
                    if any(
                        parameters[name].grad is None
                        or not torch.isfinite(parameters[name].grad).all()
                        for name in names
                    ):
                        raise ValueError("missing or nonfinite gradients")
                    gradient = torch.cat([parameters[name].grad.flatten() for name in names]).cpu()
                    save_file(
                        {"gradient": gradient, "logits": logits.detach().cpu()},
                        output / f"case-{index}.safetensors",
                    )
                    del gradient
                    gradient = None
                torch.nn.utils.clip_grad_norm_(
                    [p for group in groups for p in group["params"]], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if repeat >= config["warmups"]:
                    durations.append(elapsed)
                del logits, loss
                logits = loss = None
            if state_hash(model) != initial:
                raise ValueError("zero learning rates changed parameters")
            row.update(
                completed=True,
                input_tokens=tokens,
                measured_seconds=durations,
                median_seconds=statistics.median(durations),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                gradient_artifact_sha256=digest(output / f"case-{index}.safetensors"),
            )
        except torch.cuda.OutOfMemoryError as error:
            row.update(completed=False, failure="allocator_memory_limit", error=str(error))
        finally:
            del logits, loss, gradient
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        results.append(row)
        write(output / "progress.json", {"strategy": strategy, "cases": results})
        print(json.dumps({"strategy": strategy, **row}), flush=True)
    unchanged = state_hash(model) == initial
    if not unchanged:
        raise ValueError("profiling changed parameters")
    inputs(config)
    write(
        output / "results.json",
        {
            "strategy": strategy,
            "checkpointed_layers": checkpointed,
            "cases": results,
            "parameter_values_unchanged": unchanged,
            "parameter_order_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "configuration_sha256": digest(CONFIG),
            "initial_parameter_sha256": initial,
            "release_allowed": False,
        },
    )


def main():
    config = read(CONFIG)
    original, _ = inputs(config)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    sources = {
        str(path): digest(path)
        for path in (
            Path(__file__),
            Path("scripts/workflow_depth_execution.py"),
            Path("scripts/workflow_depth_common.py"),
            Path("scripts/probe_workflow_depth.py"),
            Path("scripts/workflow_skill_common.py"),
            Path("src/veyra/option_model.py"),
            Path("src/veyra/proper_learning.py"),
        )
    }
    write(
        OUTPUT / "protocol.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": digest(CONFIG),
            "source_sha256": sources,
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "learning_rate": 0.0,
            "calibration_or_final_used": False,
        },
    )
    for strategy in config["strategies"]:
        subprocess.run([sys.executable, "-u", __file__, "--strategy", strategy], check=True)
    reports = {name: read(OUTPUT / name / "results.json") for name in config["strategies"]}
    for key in ("configuration_sha256", "initial_parameter_sha256", "parameter_order_sha256"):
        if len({value[key] for value in reports.values()}) != 1:
            raise ValueError("strategies do not use identical initial parameters and ordering")
    comparisons = []
    for index, identifier in enumerate(original["record_ids"]):
        available = [
            name for name in config["strategies"] if reports[name]["cases"][index]["completed"]
        ]
        if not available:
            raise ValueError("no strategy completed a declared case")
        reference = available[0]
        baseline = load_file(OUTPUT / reference / f"case-{index}.safetensors")
        for name in available[1:]:
            current = load_file(OUTPUT / name / f"case-{index}.safetensors")
            difference = current["gradient"] - baseline["gradient"]
            comparison = {
                "record_id": identifier,
                "reference": reference,
                "strategy": name,
                "logits_maximum_absolute_difference": float(
                    (current["logits"] - baseline["logits"]).abs().max()
                ),
                "gradient_maximum_absolute_difference": float(difference.abs().max()),
                "gradient_relative_l2_difference": float(
                    difference.norm() / baseline["gradient"].norm().clamp_min(1e-12)
                ),
            }
            comparison["passed"] = (
                comparison["logits_maximum_absolute_difference"] <= config["logits_atol"]
                and comparison["gradient_maximum_absolute_difference"] <= config["gradient_atol"]
                and comparison["gradient_relative_l2_difference"]
                <= config["gradient_relative_l2_tolerance"]
            )
            comparisons.append(comparison)
    if not comparisons or not all(row["passed"] for row in comparisons):
        raise ValueError("declared gradient/forward parity failed")
    candidates = []
    for name, report in reports.items():
        if all(row["completed"] for row in report["cases"]):
            candidates.append((sum(row["median_seconds"] for row in report["cases"]), name))
    if not candidates:
        raise ValueError("no execution strategy completed all profiling inputs")
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError("profiling source changed")
    write(
        OUTPUT / "results.json",
        {
            "completed": True,
            "strategy_results": reports,
            "gradient_comparisons": comparisons,
            "recommended_strategy": min(candidates)[1],
            "selection": config["selection"],
            "configuration_sha256": digest(CONFIG),
            "source_sha256": sources,
            "parameter_values_unchanged": True,
            "release_allowed": False,
            "limitation": (
                "Four training inputs, one warmup and three repetitions per strategy, "
                "one device/run. Memory and speed do not bound other shapes; timings measure "
                "training steps, not inference. Where the allocator limit prevents an "
                "uncheckpointed run, parity is only between the strategies that completed "
                "that case."
            ),
        },
    )
    print(json.dumps({"completed": True, "recommended_strategy": min(candidates)[1]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=("none", "earlier12", "all24"))
    arguments = parser.parse_args()
    if arguments.strategy:
        child(arguments.strategy)
    else:
        main()
