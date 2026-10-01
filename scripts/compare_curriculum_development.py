"""Describe the completed matched curriculum study without changing selection or release."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from compare_workspace_development import paired_accuracy, parameter_difference
from curriculum_release_checks import digest, read_plain, recompute_development, require
from safetensors.torch import load_file
from train_foundation_head import write

from veyra.constants import QUESTION_TYPES
from veyra.data import TrainingRecord
from veyra.workflow_learning import annotate


def compare(output):
    config_path = Path("configs/workflow-curriculum-study-v12.json")
    selection_path = Path("runs/workflow-v12-curriculum-selection/selection.json")
    config, selection = read_plain(config_path), read_plain(selection_path)
    require(
        selection["study_protocol_sha256"] == digest(config_path)
        and selection["final_or_calibration_used"] is False,
        "comparison study identity changed",
    )
    selection_protocol = read_plain(selection_path.parent / "protocol.json")
    for name, expected_hash in selection_protocol["source_files"].items():
        require(digest(name) == expected_hash, "selection source changed: " + name)
    root = Path("runs/workflow-v12-curriculum-study")
    completed = read_plain(root / "complete.json")
    require(
        completed["completed"] is True and completed["configuration_sha256"] == digest(config_path),
        "training study is incomplete or changed",
    )
    expected = [Path(config["parent"])]
    for arm in config["arms"]:
        complete = read_plain(root / arm / "complete.json")
        require(complete["completed"] is True, "both matched arms must finish")
        expected.extend(root / arm / f"epoch-{epoch}" / "checkpoint" for epoch in (1, 2))
    candidates = selection["candidates"]
    require(
        config["arms"] == ["skill_curriculum", "outcome_replay"]
        and config["epochs"] == 2
        and len(candidates) == 5
        and [Path(row["checkpoint"]) for row in candidates] == expected,
        "require the five declared candidates",
    )
    features = Path(config["features"])
    cache = read_plain(features / "manifest.json")
    require(
        digest(features / "manifest.json") == config["feature_cache_manifest_sha256"],
        "feature cache identity changed",
    )
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        require(digest(features / name) == cache[key], "feature cache changed")
    require(digest(config["records"]) == config["records_sha256"], "source records changed")
    records = []
    for line in Path(config["records"]).open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] in {"train", "dev"}:
            records.append(TrainingRecord.model_validate(raw))
    metadata = read_plain(features / "records.json")
    annotate(metadata, records)
    tensors = load_file(features / "features.safetensors")
    indices = torch.tensor([i for i, row in enumerate(metadata) if row["split"] == "dev"])
    rows = [metadata[i] for i in indices.tolist()]
    gold, valid = tensors["gold"][indices], tensors["valid"][indices]
    predictions, states, metrics, risks, bindings = [], [], [], [], {}
    torch.set_num_threads(4)
    for row, checkpoint in zip(candidates, expected, strict=True):
        require(
            row["records_sha256"] == selection["records_sha256"],
            "candidate development corpus changed",
        )
        metric, risk = recompute_development(row, tensors, metadata, indices)
        metrics.append(metric)
        risks.append(risk)
        artifact = load_file(row["development_logits"])
        predictions.append(artifact["logits"].masked_fill(~valid, -1e9).argmax(-1))
        states.append(load_file(checkpoint / "head.safetensors"))
        bindings.update(
            {
                str(checkpoint / "head.safetensors"): row["weights_sha256"],
                str(checkpoint / "manifest.json"): row["manifest_sha256"],
                row["development_logits"]: row["development_logits_sha256"],
            }
        )
    comparisons = []
    for epoch, curriculum, control in ((1, 1, 3), (2, 2, 4)):
        differences = parameter_difference(states[0], states[curriculum], states[control])
        for row in differences.values():
            row["curriculum_update_norm"] = row.pop("primitive_update_norm")
        comparisons.append(
            {
                "epoch": epoch,
                "difference_direction": "curriculum minus matched outcome-replay control",
                "by_domain": {
                    domain: paired_accuracy(
                        predictions[control], predictions[curriculum], gold, rows, domain, 191
                    )
                    for domain in (
                        "workflow_new",
                        "workflow_known",
                        "photo_guard",
                        "text_nli",
                        "text_intent",
                    )
                },
                "by_type_coverage_under_15pct_error": {
                    kind: {
                        "curriculum": risks[curriculum][kind]["maximum_coverage_at_15pct_error"],
                        "control": risks[control][kind]["maximum_coverage_at_15pct_error"],
                    }
                    for kind in QUESTION_TYPES
                },
                "uncertainty": {
                    "curriculum": metrics[curriculum]["by_domain"]["uncertainty"],
                    "control": metrics[control]["by_domain"]["uncertainty"],
                },
                "parameter_differences": differences,
            }
        )
    write(
        output,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": digest(selection_path),
            "study_protocol_sha256": digest(config_path),
            "source_sha256": {
                **selection_protocol["source_files"],
                **{
                    str(path): digest(path)
                    for path in (
                        Path(__file__),
                        Path("scripts/compare_workspace_development.py"),
                        Path("scripts/curriculum_release_checks.py"),
                    )
                },
            },
            "bindings": bindings,
            "feature_cache": cache,
            "development_metrics_recomputed_from_raw_outputs": True,
            "bootstrap_seed": 191,
            "bootstrap_replicates": 2000,
            "comparisons": comparisons,
            "used_for_candidate_selection": False,
            "calibration_or_final_used": False,
            "release_allowed": False,
            "limitations": (
                "Descriptive merged development comparisons after the fixed selection. "
                "Accuracy intervals resample observation groups, not workflow families or "
                "training seeds, and have no multiple-comparison correction. Probability "
                "and cost values are point estimates without bootstrap intervals here. "
                "One matched training seed and repeatedly inspected development data do not "
                "establish broad robustness or independent final performance. Parameter "
                "differences do not establish causally faithful reasoning."
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = Path("runs/workflow-v12-curriculum-comparison")
    root.mkdir(parents=True, exist_ok=False)

    def record(status, **extra):
        value = {
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_allowed": False,
            **extra,
        }
        write(root / "progress.json", value)
        print(json.dumps(value), flush=True)

    try:
        if args.wait:
            record("waiting_for_merged_selection")
            while True:
                try:
                    progress = read_plain("runs/workflow-v12-curriculum-evaluation/progress.json")
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(15)
                    continue
                if progress["status"] == "failed":
                    raise RuntimeError("upstream training or selection failed; inspect its log")
                if progress["stage"] == "selection" and progress["status"] in {
                    "complete",
                    "gate_failed",
                }:
                    break
                time.sleep(15)
        record("comparing")
        compare(root / "matched-development-comparison.json")
        record("complete")
    except Exception as error:
        record("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
