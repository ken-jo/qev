"""Bind recovery weights, fitted policy and untouched validation populations."""

from datetime import datetime
from pathlib import Path

from depth_policy_checks import policy_population
from recovery_release_checks import (
    bind_source,
    collect_data,
    collect_study,
    digest,
    equal,
    read,
    require,
    study_root,
)

from veyra.constants import QUESTION_TYPES
from veyra.data import read_records

DESIGN_PATH = Path("configs/workflow-recovery-policy-v13.json")


def collect_admission(selection_path):
    study, data = collect_study(), collect_data()
    config, selection = read(DESIGN_PATH), read(selection_path)
    admission_path = Path(selection_path).parent / "admission-audit.json"
    admission = read(admission_path)
    selected = study["selected"]
    require(
        admission["passed"] is True
        and admission["selection_sha256"] == digest(selection_path)
        and admission["source_sha256"] == digest("scripts/prepare_backbone_recovery_policy.py")
        and selection["eligible"] is True
        and selection["final_or_calibration_used"] is False
        and selection["weights_sha256"] == selected["weights_sha256"]
        and selection["manifest_sha256"] == selected["manifest_sha256"]
        and selection["selected"]["checkpoint"] == selected["checkpoint"]
        and equal(selection["selected"]["recovery_metrics"], selected["metrics"])
        and equal(selection["selected"]["metrics"], selected["metrics"]["workflow"])
        and selection["study_protocol_sha256"] == digest(DESIGN_PATH)
        and selection["calibration_records_sha256"]
        == digest("data/workflow-v12-cohort/records.jsonl")
        and selection["records_sha256"] == digest(data["populations"]["final"]["records"]),
        "policy admission or development selection changed",
    )
    extension = {
        "workflow_recovery_v13": {
            "study_design_sha256": study["design_sha256"],
            "merged_selection_sha256": digest(study_root() / "selection.json"),
            "policy_design_sha256": digest(DESIGN_PATH),
            "fresh_final_data_design_sha256": config["final_data_design_sha256"],
        }
    }
    require(selection["release_extensions"] == extension, "recovery extension changed")
    evidence = {**study["evidence_files"], **data["evidence_files"]}
    for path, sha in admission["evidence_files"].items():
        bind_source(path, sha, evidence)
    for path in (DESIGN_PATH, admission_path, Path(selection_path)):
        evidence[str(path)] = digest(path)
    for role in ("validation-1", "validation-2"):
        item = selection["independent_policy_validation_records"][role]
        require(
            Path(item["path"]) == Path(data["populations"][role]["records"])
            and item["sha256"] == digest(item["path"]),
            "validation data identity changed",
        )
    require(
        config["protocol"] == "workflow-recovery-cohort-policy-v13"
        and config["weights_sha256"] == selection["weights_sha256"]
        and config["uncalibrated_manifest_sha256"] == selection["manifest_sha256"]
        and config["original_records_sha256"] == digest(config["original_records"])
        and config["method"]["policy_fit_expected_error_max"] == 0.12
        and config["method"]["policy_fit_minimum_coverage"] == 0.6
        and config["method"]["threshold_grid_denominator"] == 200,
        "declared policy or previous population changed",
    )
    return config, selection, evidence


def check_calibration(checkpoint, selection_path, calibration_report):
    config, selection, evidence = collect_admission(selection_path)
    checkpoint, calibration_report = Path(checkpoint), Path(calibration_report)
    manifest, calibration = read(checkpoint / "manifest.json"), read(calibration_report)
    weights, manifest_hash = (
        digest(checkpoint / "head.safetensors"),
        digest(checkpoint / "manifest.json"),
    )
    require(
        weights == manifest["weights_sha256"] == selection["weights_sha256"]
        and calibration["protocol"]["weights_sha256"] == weights
        and calibration["protocol"]["manifest_sha256"] == selection["manifest_sha256"]
        and calibration["calibrated_manifest_sha256"] == manifest_hash
        and calibration["calibration"] == manifest["calibration"]
        and manifest["training"]["release_extensions"] == selection["release_extensions"]
        and calibration["protocol"]["records_sha256"] == selection["calibration_records_sha256"]
        and calibration["protocol"]["selection_sha256"] == digest(selection_path)
        and calibration["protocol"]["policy_design_sha256"] == digest(DESIGN_PATH)
        and calibration["protocol"]["source_sha256"] == digest("scripts/fit_recovery_policy.py")
        and calibration["protocol"]["arguments"]["split"] == "calibration"
        and calibration["protocol"]["merged_bf16_deployment"] is True
        and manifest["training"]["intermediate"] is False,
        "calibrated recovery identity changed",
    )
    fitting = calibration["fitting"]
    require(
        set(fitting) == set(QUESTION_TYPES)
        and set(calibration["serialized_policy_check"]) == set(QUESTION_TYPES)
        and all(
            value["matches_fitted_policy"] is True
            for value in calibration["serialized_policy_check"].values()
        ),
        "serialized fitting policy failed",
    )
    for fit in fitting.values():
        require(
            fit["passed"] is True
            and fit["always_abstain"] is False
            and fit["policy_fit_expected_error_max"] == 0.12
            and fit["expected_error"] <= 0.12
            and fit["coverage"] >= 0.6
            and set(fit["by_cohort"]) == set(config["fitting_populations"])
            and all(
                value["passed"] is True
                and value["expected_error"] <= 0.12
                and value["coverage"] >= 0.6
                for value in fit["by_cohort"].values()
            ),
            "a declared fitting cohort failed",
        )
    temperature_groups = set().union(*(set(f["temperature_group_ids"]) for f in fitting.values()))
    policy_groups = set().union(*(set(f["policy_group_ids"]) for f in fitting.values()))
    fitting_records = calibration["protocol"]["arguments"]["records"]
    expected_groups = {
        r.group_id for r in read_records(Path(fitting_records)) if r.split == "calibration"
    }
    require(
        not temperature_groups & policy_groups
        and temperature_groups | policy_groups == expected_groups,
        "temperature and policy fitting populations are not separate and complete",
    )
    for path, sha in calibration["protocol"]["dependency_source_sha256"].items():
        require(digest(path) == sha, "calibration inference source changed")
        evidence[path] = sha
    predictions = calibration_report.parent / "predictions.jsonl"
    require(digest(predictions) == calibration["predictions_sha256"], "fitting outputs changed")
    for path in (
        calibration_report,
        predictions,
        checkpoint / "head.safetensors",
        checkpoint / "manifest.json",
    ):
        evidence[str(path)] = digest(path)
    return config, selection, manifest, calibration, evidence


def collect_recovery_evidence(
    checkpoint, selection_path, records, calibration_report, policy_report
):
    config, selection, manifest, calibration, evidence = check_calibration(
        checkpoint, selection_path, calibration_report
    )
    root = Path(policy_report).parent
    freeze_path = root / "policy-freeze.json"
    freeze = read(freeze_path)
    manifest_hash = digest(Path(checkpoint) / "manifest.json")
    require(
        Path(records) == Path(selection["fresh_final_records"])
        and digest(records) == selection["records_sha256"]
        and freeze["study_sha256"] == digest(DESIGN_PATH)
        and freeze["selection_sha256"] == digest(selection_path)
        and freeze["weights_sha256"] == manifest["weights_sha256"]
        and freeze["manifest_sha256"] == manifest_hash
        and freeze["calibration_report_sha256"] == digest(calibration_report)
        and freeze["validation_records"] == selection["independent_policy_validation_records"]
        and freeze["independent_validation_predictions_observed"] is False
        and freeze["source_sha256"] == digest("scripts/freeze_recovery_policy.py"),
        "policy freeze or final identity changed",
    )
    sources = freeze["source_files"].copy()
    for path, sha in sources.items():
        require(digest(path) == sha, "frozen policy source changed: " + path)
    old_path = config["previous_group_regression"]["population_report"]
    require(
        digest(old_path) == config["previous_group_regression"]["population_report_sha256"],
        "previous regression groups changed",
    )
    old = read(old_path)
    old_groups = {kind: set(value["policy_group_ids"]) for kind, value in old["fitting"].items()}
    details = {}
    for role in ("validation-1", "validation-2", "previous"):
        previous = role == "previous"
        evaluation_root = root / ("previous-calibration" if previous else role)
        report_path = Path(policy_report) if previous else root / (role + "-policy.json")
        population = (
            config["original_records"]
            if previous
            else selection["independent_policy_validation_records"][role]["path"]
        )
        current, _ = policy_population(
            evaluation_root, population, manifest, selection_path, old_groups if previous else None
        )
        report, evaluation = read(report_path), read(evaluation_root / "evaluation.json")
        require(
            report["passed"] is True
            and all(value["passed"] for value in current.values())
            and equal(report["by_type"], current)
            and report["kind"] == role
            and report["weights_sha256"] == manifest["weights_sha256"]
            and report["manifest_sha256"]
            == evaluation["protocol"]["manifest_sha256"]
            == manifest_hash
            and report["policy_freeze_sha256"] == digest(freeze_path)
            and report["evaluation_sha256"] == digest(evaluation_root / "evaluation.json")
            and report["predictions_sha256"] == digest(evaluation_root / "predictions.jsonl")
            and report["selection_sha256"] == digest(selection_path)
            and report["study_sha256"] == digest(DESIGN_PATH)
            and report["fresh_calibration_report_sha256"] == digest(calibration_report)
            and report["source_sha256"] == digest("scripts/audit_recovery_policy_population.py")
            and report["policy_refitted"] is False
            and report["population_previously_inspected"] is previous
            and report["final_predictions_used"] is False
            and datetime.fromisoformat(evaluation["protocol"]["started_at_utc"])
            >= datetime.fromisoformat(freeze["frozen_at_utc"]),
            "an independent policy population failed or changed",
        )
        if previous:
            require(
                all(
                    current[kind]["questions"] == value["questions"]
                    for kind, value in old["fitting"].items()
                ),
                "previous policy observation counts changed",
            )
        details[role] = current
        for path in (
            report_path,
            evaluation_root / "evaluation.json",
            evaluation_root / "predictions.jsonl",
        ):
            evidence[str(path)] = digest(path)
    for path in (freeze_path, Path(old_path), Path(__file__)):
        evidence[str(path)] = digest(path)
    evidence.update(sources)
    return {
        "release_extensions": selection["release_extensions"],
        "checks": {
            "recovery_study_provenance": True,
            "fresh_data_provenance": True,
            "independent_policy_validation": True,
            "previous_calibration_policy_regression": True,
        },
        "details": {
            "recovery_study_provenance": {
                "selected": selection["selected"]["checkpoint"],
                "merged_development_questions": 5102,
                "candidates": 3,
                "unfrozen_adapter_layers": list(range(18, 24)),
                "supervised_replay_and_parent_distillation": True,
                "policy_fit_error_max_per_cohort": 0.12,
            },
            "fresh_data_provenance": {
                "validation_questions_each": 2644,
                "final_questions": 3432,
                "fitting_validation_final_groups_disjoint": True,
                "procedural_targets_independently_recomputed": True,
            },
            "independent_policy_validation": {
                k: details[k] for k in ("validation-1", "validation-2")
            },
            "previous_calibration_policy_regression": details["previous"],
        },
        "evidence_files": evidence,
        "source_files": sources,
        "previous_policy_report": str(policy_report),
        "previous_policy_report_sha256": digest(policy_report),
    }
