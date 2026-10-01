"""Freeze the fitted recovery policy before either fresh validation population."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from recovery_policy_checks import DESIGN_PATH, check_calibration
from recovery_release_checks import digest, require
from train_foundation_head import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "policy freeze is immutable")
    config, selection, _, calibration, _ = check_calibration(
        args.checkpoint, args.selection, args.calibration_report
    )
    require(
        not any(
            (args.output.parent / role).exists()
            for role in (*config["validation_populations"], "baseline", "final")
        ),
        "validation or final prediction preceded policy freeze",
    )
    sources = dict(calibration["protocol"]["dependency_source_sha256"])
    for path in (
        "scripts/evaluate_workflow.py",
        "scripts/depth_policy_checks.py",
        "scripts/recovery_policy_checks.py",
        "scripts/recovery_release_checks.py",
        "scripts/prepare_backbone_recovery_policy.py",
        "scripts/fit_recovery_policy.py",
        "scripts/audit_recovery_policy_population.py",
        "scripts/freeze_recovery_policy.py",
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
    write(args.output, result)
    print(json.dumps({"policy_frozen": True, "manifest_sha256": result["manifest_sha256"]}))


if __name__ == "__main__":
    main()
