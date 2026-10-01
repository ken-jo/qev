"""Select the declared continuation candidate on unchanged development groups only."""

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
from veyra.data import read_records
from veyra.option_model import OptionModel
from veyra.workflow_learning import annotate, selection_key, summarize_logits


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config_path = Path("configs/workflow-workspace-study-v12.json")
    config = read(config_path)
    calibration_path = Path("configs/workflow-workspace-calibration-v12.json")
    calibration = read(calibration_path)
    release = read(config["release_protocol"])
    data = Path(calibration["output"]) / "records.jsonl"
    data_audit = read(data.parent / "independent-target-audit.json")
    output = Path("runs/workflow-v12-workspace-selection")
    if (output / "selection.json").exists():
        raise FileExistsError("continuation selection is frozen")
    if data_audit["passed"] is not True or data_audit["records_sha256"] != digest(data):
        raise ValueError("fresh calibration data audit is missing or stale")
    all_records = read_records(data)
    original = read_records(Path(config["records"]))
    before = [r.model_dump_json() for r in original if r.split in {"train", "dev"}]
    after = [r.model_dump_json() for r in all_records if r.split in {"train", "dev"}]
    if before != after or digest(config["records"]) != config["records_sha256"]:
        raise ValueError("training/development records changed during calibration refresh")
    records = {r.id: r for r in all_records if r.split in {"train", "dev"}}
    root = Path(config["features"])
    specification, cache = read(root / "specification.json"), read(root / "manifest.json")
    if (
        specification["records_sha256"] != config["records_sha256"]
        or specification["calibration_or_final_encoded"] is not False
        or cache["complete"] is not True
    ):
        raise ValueError("feature cache is not the completed original train/development cache")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(root / name) != cache[key]:
            raise ValueError("feature cache changed")
    tensors = load_file(root / "features.safetensors")
    rows = read(root / "records.json")
    if len(rows) != len(records) or {row["id"] for row in rows} != set(records):
        raise ValueError("feature cache population changed")
    annotate(rows, records.values())
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    candidates = [Path(config["parent"])]
    for arm in config["execution_order"]:
        run = Path("runs/workflow-v12-workspace-study") / arm
        complete = read(run / "complete.json")
        protocol = read(run / "protocol.json")
        if complete["completed"] is not True or protocol["config_sha256"] != digest(config_path):
            raise ValueError("continuation arm is incomplete or used another study")
        if protocol["feature_cache"] != cache or protocol["calibration_or_final_used"] is not False:
            raise ValueError("continuation used another cache or a held-out split")
        for name, expected_hash in protocol["source_sha256"].items():
            if digest(name) != expected_hash:
                raise ValueError("continuation source changed: " + name)
        expected = [run / f"epoch-{i}" / "checkpoint" for i in range(1, config["epochs"] + 1)]
        if list(map(Path, complete["candidates"])) != expected:
            raise ValueError("undeclared continuation checkpoint")
        candidates.extend(expected)
    if len(candidates) != 5:
        raise ValueError("require parent plus both epochs of both declared arms")
    protocol = {
        "study_protocol_sha256": digest(config_path),
        "calibration_design_sha256": digest(calibration_path),
        "release_protocol_sha256": digest(config["release_protocol"]),
        "records_sha256": digest(data),
        "data_audit_sha256": digest(data.parent / "independent-target-audit.json"),
        "feature_cache": cache,
        "candidates": [str(path) for path in candidates],
        "source_files": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("scripts/train_workflow_workspace.py"),
                Path("scripts/workflow_workspace_supervision.py"),
                Path("scripts/workspace_development_diagnostics.py"),
                Path("src/veyra/workflow_learning.py"),
            )
        },
        "calibration_or_final_used_for_selection": False,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        prior = read(protocol_path)
        if {k: v for k, v in prior.items() if k != "started_at_utc"} != protocol:
            raise ValueError("resumed selection protocol changed")
    else:
        write(protocol_path, {**protocol, "started_at_utc": datetime.now(timezone.utc).isoformat()})
    torch.set_num_threads(4)

    def measure(checkpoint, path):
        weights, manifest = (
            digest(checkpoint / "head.safetensors"),
            digest(checkpoint / "manifest.json"),
        )
        if path.exists():
            result = read(path)
            if (result["weights_sha256"], result["manifest_sha256"], result["records_sha256"]) != (
                weights,
                manifest,
                digest(data),
            ):
                raise ValueError("resumed development evidence changed")
            if digest(result["development_logits"]) != result["development_logits_sha256"]:
                raise ValueError("resumed development logits changed")
            return result
        model = OptionModel.load(checkpoint, local_files_only=True, merge=True)
        model.eval()
        values = []
        with torch.inference_mode():
            for i in dev.tolist():
                logits, _ = forward(model, records[rows[i]["id"]], data.parent)
                values.append(logits[0].cpu())
        logits = torch.stack(values)
        temperatures = model.calibration.temperatures
        measured = {
            "metrics": summarize_logits(logits, tensors, rows, dev, temperatures),
            "development_risk_diagnostic": development_risk(logits, tensors, dev, temperatures),
            "additional_development_diagnostics": diagnose(
                logits, tensors, rows, dev, records, temperatures
            ),
        }
        logits_path = path.with_suffix(".safetensors")
        save_file({"logits": logits.contiguous(), "cache_indices": dev}, logits_path)
        result = {
            "checkpoint": str(checkpoint),
            "weights_sha256": weights,
            "manifest_sha256": manifest,
            "records_sha256": digest(data),
            "development_logits": str(logits_path),
            "development_logits_sha256": digest(logits_path),
            **measured,
        }
        write(path, result)
        del model, logits, values
        gc.collect()
        torch.cuda.empty_cache()
        return result

    baseline_path = Path(release["baseline"])
    if (
        digest(baseline_path / "head.safetensors") != release["baseline_weights_sha256"]
        or digest(baseline_path / "manifest.json") != release["baseline_manifest_sha256"]
    ):
        raise ValueError("frozen Foundation changed")
    baseline_report = output / "merged-baseline-dev.json"
    baseline = measure(baseline_path, baseline_report)
    results = []
    for index, checkpoint in enumerate(candidates):
        result = measure(checkpoint, output / f"merged-dev-{index}.json")
        primary = selection_key(result["metrics"], baseline["metrics"])
        risk = result["development_risk_diagnostic"]
        passes_risk = set(risk) == set(QUESTION_TYPES) and all(
            row["maximum_coverage_at_15pct_error"] >= 0.6 for row in risk.values()
        )
        result["primary_selection"] = primary
        result["development_abstention_passed"] = passes_risk
        result["selection"] = (bool(primary[0] and passes_risk), *primary)
        results.append(result)
        print(
            json.dumps({"candidate": index, "selection": result["selection"], "risk": risk}),
            flush=True,
        )
    selected = max(results, key=lambda result: tuple(result["selection"]))
    result = {
        "selected": selected,
        "candidates": results,
        "weights_sha256": selected["weights_sha256"],
        "manifest_sha256": selected["manifest_sha256"],
        "baseline_merged_dev": str(baseline_report),
        "baseline_merged_dev_sha256": digest(baseline_report),
        "records_sha256": digest(data),
        "release_protocol_sha256": digest(config["release_protocol"]),
        "study_protocol_sha256": digest(config_path),
        "calibration_design_sha256": digest(calibration_path),
        "eligible": bool(selected["selection"][0]),
        "required_priorities": [1, 2],
        "final_or_calibration_used": False,
        "prior_calibration_failure_reviewed_as_motivation": True,
        "release_extensions": {
            "workflow_workspace_v12": {
                "study_protocol_sha256": digest(config_path),
                "calibration_design_sha256": digest(calibration_path),
            }
        },
    }
    write(output / "selection.json", result)
    print(
        json.dumps({"selected": selected["checkpoint"], "eligible": result["eligible"]}), flush=True
    )


if __name__ == "__main__":
    main()
