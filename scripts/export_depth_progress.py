"""Preserve completed depth epochs without treating interim development as release evidence."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def verify_index(output):
    index = read(output / "manifest.json")
    for name, expected in index.items():
        if Path(name).name != name or digest(output / name) != expected:
            raise ValueError("existing public evidence changed: " + name)
    return index


def export(config_path, source, output):
    config = read(config_path)
    config_sha = digest(config_path)
    index = verify_index(output)
    exported = []
    for arm in config["arms"]:
        protocol_path = source / arm / "protocol.json"
        manifests = [
            source / arm / f"epoch-{epoch}" / "checkpoint" / "manifest.json"
            for epoch in range(1, config["epochs"] + 1)
        ]
        if not any(path.exists() for path in manifests):
            continue
        protocol = read(protocol_path)
        if not (
            protocol["configuration_sha256"] == config_sha
            and protocol["arm"] == arm
            and protocol["training_plan_sha256"] == config["training_plan_sha256"]
            and protocol["calibration_or_final_used"] is False
        ):
            raise ValueError("training protocol differs from the fixed declaration")
        for name, expected in protocol["source_sha256"].items():
            if digest(name) != expected:
                raise ValueError("training source changed: " + name)
        initialization_path = source / arm / "initialization.json"
        initialization = read(initialization_path)
        if not (
            initialization["configuration_sha256"] == config_sha
            and initialization["earlier_learning_rate_multiplier"]
            == config["earlier_learning_rate_multiplier"][arm]
        ):
            raise ValueError("initialization differs from the fixed declaration")
        comparison = {"role": "reference", "control_comparison_performed": False}
        if arm != "full_depth":
            reference_path = source / "full_depth" / "initialization.json"
            reference = read(reference_path)
            for key in (
                "trainable_parameter_sha256",
                "parameter_group_sha256",
                "parameter_count_by_group",
                "expansion",
                "configuration_sha256",
            ):
                if initialization[key] != reference[key]:
                    raise ValueError("depth arms have different initialization: " + key)
            comparison = {
                "role": "control",
                "control_comparison_performed": True,
                "matches_reference": True,
                "reference_initialization_sha256": digest(reference_path),
            }
        for epoch, manifest_path in enumerate(manifests, start=1):
            if not manifest_path.exists():
                continue
            manifest = read(manifest_path)
            training = manifest["training"]
            weights_sha = digest(manifest_path.parent / "head.safetensors")
            if not (
                training["completed"] is True
                and training["intermediate"] is True
                and training["method"] == "depth-" + arm
                and training["epochs"] == epoch
                and training["seed"] == config["seed"]
                and training["protocol_sha256"] == digest(protocol_path)
                and training["initialization_sha256"] == digest(initialization_path)
                and training["parent_weights_sha256"] == config["parent_weights_sha256"]
                and training["dataset_sha256"] == config["records_sha256"]
                and training["earlier_adapters_updated"] == (arm == "full_depth")
                and manifest["weights_sha256"] == weights_sha
                and training["final_evaluated_at_selection"] is False
                and training["smoke"] is False
            ):
                raise ValueError("completed epoch evidence does not match its checkpoint")
            report = {
                "status": "interim_unmerged_development",
                "arm": arm,
                "epoch": epoch,
                "checkpoint": manifest_path.parent.as_posix(),
                "weights_sha256": weights_sha,
                "manifest_sha256": digest(manifest_path),
                "configuration_sha256": config_sha,
                "exporter_sha256": digest(__file__),
                "training_protocol": protocol,
                "initialization": initialization,
                "initialization_comparison": comparison,
                "training": training,
                "release_allowed": False,
                "limitation": (
                    "Repeatedly inspected development data and one training seed. "
                    "These are unmerged checkpoint diagnostics. Merged selection, calibration "
                    "and final evaluation remain separate. Skill-query metrics cannot "
                    "substitute for the original release criteria."
                ),
            }
            target = output / f"{arm}-epoch-{epoch}.json"
            if target.exists():
                if read(target) != report:
                    raise ValueError("existing epoch aggregate changed: " + str(target))
            else:
                write(target, report)
            index[target.name] = digest(target)
            domains = training["selected_dev"]["by_domain"]
            exported.append(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "new_accuracy": domains["workflow_new"]["accuracy"],
                    "known_accuracy": domains["workflow_known"]["accuracy"],
                    "uncertainty_nll": domains["uncertainty"]["nll"],
                    "coverage_at_15pct_error": {
                        kind: row["maximum_coverage_at_15pct_error"]
                        for kind, row in training["development_risk_diagnostic"].items()
                    },
                }
            )
    if exported:
        write(output / "manifest.json", dict(sorted(index.items())))
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    config_path = Path("configs/workflow-depth-study-v12.json")
    config = read(config_path)
    source = Path("runs/workflow-v12-depth-study")
    output = Path("reports/workflow-v12/depth-study")
    expected_count = len(config["arms"]) * config["epochs"]
    previous = None
    exported = []
    while True:
        # Checkpoint manifests appear by atomic rename after weights have been saved.
        present = tuple(
            (arm, epoch)
            for arm in config["arms"]
            for epoch in range(1, config["epochs"] + 1)
            if (source / arm / f"epoch-{epoch}" / "checkpoint" / "manifest.json").exists()
        )
        if present != previous:
            exported = export(config_path, source, output)
            status = "epochs_exported" if exported else "pending_first_completed_epoch"
            print(
                json.dumps(
                    {"status": status, "exported": exported, "release_allowed": False}, indent=2
                ),
                flush=True,
            )
            previous = present
        if not args.wait or len(exported) == expected_count:
            return
        if (source / "complete.json").exists():
            # Re-scan once: the last checkpoint may have appeared during this iteration.
            if len(export(config_path, source, output)) != expected_count:
                raise ValueError("training reports completion but declared checkpoints are missing")
            return
        try:
            progress = read(source / "progress.json")
        except (FileNotFoundError, json.JSONDecodeError):
            # Root progress is not a checkpoint and may be observed during its short write.
            time.sleep(1)
            progress = read(source / "progress.json")
        if progress["status"] == "failed":
            raise RuntimeError("training failed; exported aggregates remain preserved")
        time.sleep(15)


if __name__ == "__main__":
    main()
