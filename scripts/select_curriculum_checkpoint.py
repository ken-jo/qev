"""Merged selection for the predeclared skill-curriculum versus replay study."""

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from train_foundation_head import write
from train_workflow_backbone import forward
from train_workflow_workspace import development_risk
from workspace_development_diagnostics import diagnose

from veyra.constants import QUESTION_TYPES
from veyra.data import TrainingRecord
from veyra.option_model import OptionModel
from veyra.workflow_learning import annotate, selection_key, summarize_logits


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_development(path):
    result = {}
    for line in Path(path).open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] in {"train", "dev"}:
            record = TrainingRecord.model_validate(raw)
            result[record.id] = record
    return result


def main():
    config_path = Path("configs/workflow-curriculum-study-v12.json")
    config = read(config_path)
    output = Path("runs/workflow-v12-curriculum-selection")
    if (output / "selection.json").exists():
        raise FileExistsError("curriculum selection already finished")
    root = Path("runs/workflow-v12-curriculum-study")
    complete = read(root / "complete.json")
    if not complete["completed"] or complete["configuration_sha256"] != digest(config_path):
        raise ValueError("the declared training study is incomplete")
    design_path = Path(config["fresh_calibration_design"])
    design = read(design_path)
    data = Path(design["output"]) / "records.jsonl"
    data_audit = read(data.parent / "independent-target-audit.json")
    if not data_audit["passed"] or data_audit["records_sha256"] != digest(data):
        raise ValueError("fresh calibration data audit changed")
    bindings = {
        config_path: digest(config_path),
        Path(config["release_protocol"]): config["release_protocol_sha256"],
        Path(config["records"]): config["records_sha256"],
        design_path: config["fresh_calibration_design_sha256"],
        Path(config["parent"]) / "head.safetensors": config["parent_weights_sha256"],
        Path(config["parent"]) / "manifest.json": config["parent_manifest_sha256"],
        Path(config["training_plan"]): config["training_plan_sha256"],
        Path(config["skills"]) / "audit.json": config["skills_audit_sha256"],
        Path(config["skills"]) / "independent-target-audit.json": config[
            "skills_independent_audit_sha256"
        ],
    }
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError("declared selection input changed: " + str(path))
    records = training_development(config["records"])
    fresh_records = training_development(data)
    if records != fresh_records:
        raise ValueError("fresh calibration construction changed training/development")
    features = Path(config["features"])
    cache = read(features / "manifest.json")
    if digest(features / "manifest.json") != config["feature_cache_manifest_sha256"]:
        raise ValueError("cache identity changed")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(features / name) != cache[key]:
            raise ValueError("cache content changed")
    rows, tensors = read(features / "records.json"), load_file(features / "features.safetensors")
    if {row["id"] for row in rows} != set(records):
        raise ValueError("development cache population changed")
    annotate(rows, records.values())
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    candidates = [Path(config["parent"])]
    sources = {}
    for arm in config["arms"]:
        run = root / arm
        training, completed = read(run / "protocol.json"), read(run / "complete.json")
        if (
            not completed["completed"]
            or training["configuration_sha256"] != digest(config_path)
            or training["calibration_or_final_used"]
        ):
            raise ValueError("incomplete or undeclared training arm")
        if (
            training["feature_cache"] != cache
            or training["training_plan_sha256"] != config["training_plan_sha256"]
        ):
            raise ValueError("training used different data or schedule")
        expected = [
            run / f"epoch-{epoch}" / "checkpoint" for epoch in range(1, config["epochs"] + 1)
        ]
        if list(map(Path, completed["candidates"])) != expected:
            raise ValueError("candidate pool changed")
        for path, source_hash in training["source_sha256"].items():
            if digest(path) != source_hash:
                raise ValueError("training source changed: " + path)
            sources[path] = source_hash
        for checkpoint in expected:
            if read(checkpoint / "manifest.json")["training"]["protocol_sha256"] != digest(
                run / "protocol.json"
            ):
                raise ValueError("candidate training identity mismatch")
        candidates.extend(expected)
    if len(candidates) != 5:
        raise ValueError("require the five predeclared candidates")
    prior_path = Path("runs/workflow-v12-workspace-selection/selection.json")
    prior = read(prior_path)
    if digest(prior_path) != config["prior_selection_sha256"] or prior["eligible"]:
        raise ValueError("prior failed selection changed")
    prior_protocol = read(prior_path.parent / "protocol.json")
    for path, expected in prior_protocol["source_files"].items():
        if digest(path) != expected:
            raise ValueError("reused baseline inference source changed")
    baseline = read(config["baseline_development"])
    if digest(config["baseline_development"]) != config["baseline_development_sha256"]:
        raise ValueError("frozen merged baseline changed")
    parent_report = prior["candidates"][2]
    if Path(parent_report["checkpoint"]) != candidates[0]:
        raise ValueError("cached parent identity mismatch")
    for report in (baseline, parent_report):
        checkpoint = Path(report["checkpoint"])
        if (
            report["weights_sha256"] != digest(checkpoint / "head.safetensors")
            or report["manifest_sha256"] != digest(checkpoint / "manifest.json")
            or report["records_sha256"] != digest(data)
        ):
            raise ValueError("reused merged measurements changed")
        artifact = load_file(report["development_logits"])
        if digest(report["development_logits"]) != report[
            "development_logits_sha256"
        ] or not torch.equal(artifact["cache_indices"], dev):
            raise ValueError("reused logits or development row order changed")
    sources.update(
        {
            str(path): digest(path)
            for path in (Path(__file__), Path("scripts/workspace_development_diagnostics.py"))
        }
    )
    protocol = {
        "study_protocol_sha256": digest(config_path),
        "release_protocol_sha256": config["release_protocol_sha256"],
        "calibration_design_sha256": digest(design_path),
        "records_sha256": digest(data),
        "data_audit_sha256": digest(data.parent / "independent-target-audit.json"),
        "feature_cache": cache,
        "candidates": [str(path) for path in candidates],
        "source_files": sources,
        "reused_merged_evidence": {
            "baseline_report_sha256": config["baseline_development_sha256"],
            "prior_selection_sha256": config["prior_selection_sha256"],
            "parent_logits_sha256": parent_report["development_logits_sha256"],
            "reason": "Identical data, ordering, weights, manifests and inference sources.",
        },
        "calibration_or_final_used_for_selection": False,
    }
    if (output / "protocol.json").exists():
        old = read(output / "protocol.json")
        if {key: value for key, value in old.items() if key != "started_at_utc"} != protocol:
            raise ValueError("resumed selection changed")
    else:
        write(
            output / "protocol.json",
            {**protocol, "started_at_utc": datetime.now(timezone.utc).isoformat()},
        )
    torch.set_num_threads(4)
    results = []
    for index, checkpoint in enumerate(candidates):
        report_path = output / f"merged-dev-{index}.json"
        if index == 0:
            result = {
                key: value
                for key, value in parent_report.items()
                if key not in {"selection", "primary_selection", "development_abstention_passed"}
            }
        elif report_path.exists():
            result = read(report_path)
            if (
                result["weights_sha256"] != digest(checkpoint / "head.safetensors")
                or result["manifest_sha256"] != digest(checkpoint / "manifest.json")
                or result["records_sha256"] != digest(data)
                or result["development_logits_sha256"] != digest(result["development_logits"])
            ):
                raise ValueError("resumed candidate changed")
        else:
            model = OptionModel.load(checkpoint, local_files_only=True, merge=True)
            model.eval()
            with torch.inference_mode():
                logits = torch.stack(
                    [
                        forward(model, records[rows[i]["id"]], data.parent)[0][0].cpu()
                        for i in dev.tolist()
                    ]
                )
            logits_path = report_path.with_suffix(".safetensors")
            save_file({"logits": logits, "cache_indices": dev}, logits_path)
            result = {
                "checkpoint": str(checkpoint),
                "weights_sha256": digest(checkpoint / "head.safetensors"),
                "manifest_sha256": digest(checkpoint / "manifest.json"),
                "records_sha256": digest(data),
                "development_logits": str(logits_path),
                "development_logits_sha256": digest(logits_path),
                "metrics": summarize_logits(
                    logits, tensors, rows, dev, model.calibration.temperatures
                ),
                "development_risk_diagnostic": development_risk(
                    logits, tensors, dev, model.calibration.temperatures
                ),
                "additional_development_diagnostics": diagnose(
                    logits, tensors, rows, dev, records, model.calibration.temperatures
                ),
            }
            del model, logits
            gc.collect()
            torch.cuda.empty_cache()
        primary = selection_key(result["metrics"], baseline["metrics"])
        risk = result["development_risk_diagnostic"]
        passes = set(risk) == set(QUESTION_TYPES) and all(
            row["maximum_coverage_at_15pct_error"] >= 0.6 for row in risk.values()
        )
        result.update(
            primary_selection=primary,
            development_abstention_passed=passes,
            selection=(bool(primary[0] and passes), *primary),
        )
        write(report_path, result)
        results.append(result)
        print(
            json.dumps({"candidate": index, "selection": result["selection"], "risk": risk}),
            flush=True,
        )
    selected = max(results, key=lambda row: tuple(row["selection"]))
    report = {
        "selected": selected,
        "candidates": results,
        "weights_sha256": selected["weights_sha256"],
        "manifest_sha256": selected["manifest_sha256"],
        "baseline_merged_dev": config["baseline_development"],
        "baseline_merged_dev_sha256": config["baseline_development_sha256"],
        "records_sha256": digest(data),
        "release_protocol_sha256": config["release_protocol_sha256"],
        "study_protocol_sha256": digest(config_path),
        "calibration_design_sha256": digest(design_path),
        "eligible": bool(selected["selection"][0]),
        "required_priorities": [1, 2],
        "final_or_calibration_used": False,
        "release_extensions": {
            "workflow_curriculum_v12": {
                "study_protocol_sha256": digest(config_path),
                "calibration_design_sha256": digest(design_path),
            }
        },
    }
    write(output / "selection.json", report)
    print(
        json.dumps({"selected": selected["checkpoint"], "eligible": report["eligible"]}), flush=True
    )


if __name__ == "__main__":
    main()
