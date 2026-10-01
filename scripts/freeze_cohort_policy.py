"""Freeze the fitted policy before any independent validation prediction."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from cohort_policy_checks import DESIGN_PATH, collect_cohort_study
from depth_policy_checks import digest, read, require

from veyra.constants import QUESTION_TYPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    require(not args.output.exists(), "policy freeze already exists")
    config = read(DESIGN_PATH)
    fit_design = read(config["fitting_data_design"])
    collect_cohort_study(args.selection, Path(fit_design["output"]) / "records.jsonl")
    selection, calibration = read(args.selection), read(args.calibration_report)
    manifest = read(args.checkpoint / "manifest.json")
    require(
        digest(args.checkpoint / "head.safetensors")
        == selection["weights_sha256"]
        == calibration["protocol"]["weights_sha256"]
        == manifest["weights_sha256"]
        and digest(args.checkpoint / "manifest.json") == calibration["calibrated_manifest_sha256"]
        and calibration["protocol"]["policy_design_sha256"] == digest(DESIGN_PATH)
        and calibration["protocol"]["selection_sha256"] == digest(args.selection)
        and calibration["protocol"]["source_sha256"] == digest("scripts/fit_cohort_policy.py")
        and calibration["calibration"] == manifest["calibration"]
        and set(calibration["fitting"]) == set(QUESTION_TYPES)
        and all(
            value["passed"] is True
            and value["always_abstain"] is False
            and value["policy_fit_expected_error_max"] == 0.12
            and value["expected_error"] <= 0.12
            and value["coverage"] >= 0.6
            and set(value["by_cohort"]) == set(config["fitting_populations"])
            and all(
                cohort["passed"] is True
                and cohort["expected_error"] <= 0.12
                and cohort["coverage"] >= 0.6
                for cohort in value["by_cohort"].values()
            )
            for value in calibration["fitting"].values()
        )
        and set(calibration["serialized_policy_check"]) == set(QUESTION_TYPES)
        and all(
            value["matches_fitted_policy"] is True
            for value in calibration["serialized_policy_check"].values()
        ),
        "conservative calibration is incomplete or changed",
    )
    require(
        not any((args.output.parent / role).exists() for role in config["validation_populations"]),
        "validation started before freeze",
    )
    require(
        calibration["protocol"]["records_sha256"] == selection["records_sha256"]
        and calibration["protocol"]["manifest_sha256"] == selection["manifest_sha256"]
        and calibration["protocol"]["arguments"]["split"] == "calibration"
        and manifest["training"]["release_extensions"] == selection["release_extensions"],
        "fitting records or model lineage changed",
    )
    sources = dict(calibration["protocol"]["dependency_source_sha256"])
    for path, expected in sources.items():
        require(digest(path) == expected, "calibration inference source changed")
    for path in (
        "scripts/evaluate_workflow.py",
        "scripts/depth_policy_checks.py",
        "scripts/cohort_policy_checks.py",
        "scripts/fit_cohort_policy.py",
        "scripts/audit_cohort_policy_population.py",
        "scripts/freeze_cohort_policy.py",
    ):
        sources[path] = digest(path)
    result = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(__file__),
        "source_files": sources,
        "study_sha256": digest(DESIGN_PATH),
        "selection_sha256": digest(args.selection),
        "weights_sha256": digest(args.checkpoint / "head.safetensors"),
        "manifest_sha256": digest(args.checkpoint / "manifest.json"),
        "calibration_report_sha256": digest(args.calibration_report),
        "validation_records": selection["independent_policy_validation_records"],
        "independent_validation_predictions_observed": False,
        "release_allowed": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"policy_frozen": True, "manifest_sha256": result["manifest_sha256"]}))


if __name__ == "__main__":
    main()
