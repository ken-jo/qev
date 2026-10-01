"""Admit only the fully audited merged recovery selection to the predeclared policy fit."""

import json
from datetime import datetime, timezone
from pathlib import Path

from recovery_release_checks import collect_data, collect_study, digest, read, require, study_root
from train_foundation_head import write


def main():
    output = Path("runs/backbone-recovery-policy-selection-v13")
    policy_path = Path("configs/workflow-recovery-policy-v13.json")
    require(not output.exists() and not policy_path.exists(), "policy admission is immutable")
    study, data = collect_study(), collect_data()
    design = read("configs/workflow-backbone-recovery-v13.json")
    release = read(design["followup_design"])
    require(
        digest(design["followup_design"]) == design["followup_design_sha256"],
        "prospective follow-up design changed",
    )
    old_policy = read("configs/workflow-cohort-policy-v12.json")
    selected = study["selected"]
    merged_selection = study_root() / "selection.json"
    policy = {
        "protocol": "workflow-recovery-cohort-policy-v13",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_declared_before_training_in": design["followup_design"],
        "method_declaration_sha256": design["followup_design_sha256"],
        "checkpoint": selected["checkpoint"],
        "weights_sha256": selected["weights_sha256"],
        "uncalibrated_manifest_sha256": selected["manifest_sha256"],
        "release_protocol": design["release_protocol"],
        "release_protocol_sha256": design["release_protocol_sha256"],
        "fitting_data_design": old_policy["fitting_data_design"],
        "fitting_data_design_sha256": old_policy["fitting_data_design_sha256"],
        "fitting_populations": old_policy["fitting_populations"],
        "method": old_policy["method"],
        "validation_populations": {
            f"validation-{index}": spec
            for index, spec in enumerate(release["validation_populations"], start=1)
        },
        "previous_group_regression": old_policy["previous_group_regression"],
        "original_records": "data/workflow-v12/records.jsonl",
        "original_records_sha256": digest("data/workflow-v12/records.jsonl"),
        "final_data_design": release["final_data_design"],
        "final_data_design_sha256": release["final_data_design_sha256"],
        "fresh_validation_or_final_predictions_observed": False,
        "calibration_populations_previously_inspected": True,
        "release_allowed": False,
    }
    require(
        policy["method"]["policy_fit_expected_error_max"] == 0.12
        and policy["method"]["policy_fit_minimum_coverage"] == 0.6
        and policy["method"]["threshold_grid_denominator"] == 200,
        "declared fitting rule changed",
    )
    for name in ("validation-1", "validation-2", "final", "calibration"):
        require(
            not (Path("runs/backbone-recovery-release-v13") / name).exists(),
            "policy admission follows recovery inference",
        )
    write(policy_path, policy)
    extension = {
        "workflow_recovery_v13": {
            "study_design_sha256": digest("configs/workflow-backbone-recovery-v13.json"),
            "merged_selection_sha256": digest(merged_selection),
            "policy_design_sha256": digest(policy_path),
            "fresh_final_data_design_sha256": release["final_data_design_sha256"],
        }
    }
    baseline = Path("runs/workflow-v12-workspace-selection/merged-baseline-dev.json")
    selection = {
        "selected": {
            **selected,
            "metrics": selected["metrics"]["workflow"],
            "recovery_metrics": selected["metrics"],
        },
        "weights_sha256": selected["weights_sha256"],
        "manifest_sha256": selected["manifest_sha256"],
        "baseline_merged_dev": str(baseline),
        "baseline_merged_dev_sha256": digest(baseline),
        "records_sha256": digest(data["populations"]["final"]["records"]),
        "calibration_records_sha256": release["calibration_records_sha256"],
        "release_protocol_sha256": design["release_protocol_sha256"],
        "study_protocol_sha256": digest(policy_path),
        "release_extensions": extension,
        "eligible": True,
        "final_or_calibration_used": False,
        "merged_recovery_selection": str(merged_selection),
        "merged_recovery_selection_sha256": digest(merged_selection),
        "independent_policy_validation_records": {
            role: {
                "path": data["populations"][role]["records"],
                "sha256": digest(data["populations"][role]["records"]),
            }
            for role in ("validation-1", "validation-2")
        },
        "fresh_final_records": data["populations"]["final"]["records"],
    }
    write(output / "selection.json", selection)
    write(
        output / "admission-audit.json",
        {
            "passed": True,
            "weights_sha256": selected["weights_sha256"],
            "source_sha256": digest(Path(__file__)),
            "selection_sha256": digest(output / "selection.json"),
            "recomputed_candidate_count": 3,
            "development_questions_per_candidate": 5102,
            "evidence_files": {
                **study["evidence_files"],
                **data["evidence_files"],
                str(policy_path): digest(policy_path),
            },
            "release_allowed": False,
        },
    )
    print(json.dumps({"admitted": selected["checkpoint"], "release_allowed": False}), flush=True)


if __name__ == "__main__":
    main()
