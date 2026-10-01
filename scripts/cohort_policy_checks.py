"""Audit adaptive cohort fitting and two prospectively declared policy validations."""

from datetime import datetime
from pathlib import Path

from depth_policy_checks import (
    collect_policy_design as collect_prior_design,
)
from depth_policy_checks import (
    digest,
    normalized_state,
    policy_population,
    read,
    require,
)

from veyra.constants import QUESTION_TYPES
from veyra.data import read_records

DESIGN_PATH = Path("configs/workflow-cohort-policy-v12.json")
REPAIR_PATH = Path("configs/workflow-cohort-policy-data-repair-v12.json")


def collect_cohort_design():
    prior = collect_prior_design()
    config = read(DESIGN_PATH)
    original = prior["original_selection"]
    evidence, sources = prior["evidence_files"], prior["source_files"]
    require(
        config["protocol"] == "workflow-cohort-robust-policy-v12"
        and config["prior_policy_study_sha256"] == digest(config["prior_policy_study"])
        and config["development_selection_sha256"] == digest(config["development_selection"])
        and config["weights_sha256"] == original["weights_sha256"]
        and config["uncalibrated_manifest_sha256"] == original["manifest_sha256"]
        and config["release_protocol_sha256"] == original["release_protocol_sha256"]
        and config["required_priorities"] == [1, 2],
        "cohort study changed original weight selection or acceptance criteria",
    )
    require(
        config["method"]["name"] == "cohort_robust_empirical_policy"
        and config["method"]["policy_fit_expected_error_max"] == 0.12
        and config["method"]["policy_fit_minimum_coverage"] == 0.6
        and config["method"]["threshold_grid_denominator"] == 200
        and config["method"]["one_policy_only"] is True
        and config["method"]["candidate_weights_reselected"] is False
        and config["independent_validation"]["both_populations_must_pass"] is True,
        "cohort method changed",
    )
    require(
        set(config["fitting_populations"]) == {"b", "c", "d"}
        and set(config["validation_populations"]) == {"validation-1", "validation-2"},
        "required fitting or validation population is missing",
    )
    for path, expected in config["source_files"].items():
        require(digest(path) == expected, "declared cohort source changed: " + path)
        sources[path] = expected
    for key in ("failure", "diagnosis"):
        path = config["trigger"][key]
        require(digest(path) == config["trigger"][key + "_sha256"], "prior failure changed")
        evidence[path] = digest(path)
    require(read(config["trigger"]["failure"])["passed"] is False, "prior validation did not fail")
    repair = read(REPAIR_PATH)
    for key in ("failed_source", "replacement_source", "failure_report"):
        require(digest(repair[key]) == repair[key + "_sha256"], "data repair evidence changed")
        evidence[repair[key]] = digest(repair[key])
    require(
        repair["study_sha256"] == digest(DESIGN_PATH)
        and repair["method_or_sampling_changed"] is False
        and repair["cohort_policy_fitting_or_validation_predictions_observed"] is False,
        "data repair changed the declared experiment",
    )
    sources[repair["replacement_source"]] = digest(repair["replacement_source"])
    design_path = Path(config["fitting_data_design"])
    require(digest(design_path) == config["fitting_data_design_sha256"], "fitting design changed")
    fit_path = Path(read(design_path)["output"]) / "records.jsonl"
    protocol = read(fit_path.parent / "protocol.json")
    require(
        protocol["passed"] is True
        and protocol["records_sha256"] == digest(fit_path)
        and protocol["study_sha256"] == digest(DESIGN_PATH)
        and protocol["design_sha256"] == digest(design_path)
        and protocol["repair_protocol_sha256"] == digest(REPAIR_PATH)
        and protocol["source_sha256"] == repair["replacement_source_sha256"],
        "combined fitting data is not the audited repaired output",
    )
    original_records = read_records(Path(read(config["origin_study"])["records"]))
    records = read_records(fit_path)
    calibration = [r for r in records if r.split == "calibration"]
    source_records, cohort_groups = [], {}
    for cohort, spec in config["fitting_populations"].items():
        require(digest(spec["records"]) == spec["records_sha256"], "cohort records changed")
        rows = [r for r in read_records(Path(spec["records"])) if r.split == "calibration"]
        source_records.extend(rows)
        cohort_groups[cohort] = {r.group_id for r in rows}
        evidence[spec["records"]] = digest(spec["records"])
    require(
        [r.model_dump_json() for r in calibration] == [r.model_dump_json() for r in source_records]
        and len(calibration) == 7932
        and len({r.group_id for r in calibration}) == 3252,
        "fitting data does not exactly combine the three declared cohorts",
    )
    datasets = {"fit": fit_path}
    previous = list(original_records) + calibration
    for role, spec in config["validation_populations"].items():
        data_design = read(spec["design"])
        path = Path(spec["records"])
        require(
            digest(spec["design"]) == spec["design_sha256"]
            and path == Path(data_design["output"]) / "records.jsonl",
            "validation design changed",
        )
        data_protocol, audit = (
            read(path.parent / "protocol.json"),
            read(path.parent / "independent-target-audit.json"),
        )
        require(
            data_protocol["records_sha256"] == audit["records_sha256"] == digest(path)
            and data_protocol["calibration_design_sha256"]
            == audit["calibration_design_sha256"]
            == digest(spec["design"])
            and data_protocol["study_protocol_sha256"] == digest(DESIGN_PATH)
            and data_protocol["generator_sha256"]
            == sources["scripts/build_followup_calibration.py"]
            and audit["source_sha256"] == sources["scripts/verify_followup_calibration_data.py"]
            and data_protocol["model_outputs_used"] is False
            and audit["passed"] is True
            and audit["fresh_policy_targets_recomputed"] == 2040
            and audit["maximum_target_absolute_error"] <= 1e-7
            and datetime.fromisoformat(data_protocol["created_at_utc"])
            > datetime.fromisoformat(config["declared_at_utc"]),
            "validation target or prospective data audit failed",
        )
        current = read_records(path)
        for split in ("train", "dev", "test"):
            require(
                [r.model_dump_json() for r in original_records if r.split == split]
                == [r.model_dump_json() for r in current if r.split == split]
                == [r.model_dump_json() for r in records if r.split == split],
                "original non-calibration records changed",
            )
        fresh = [r for r in current if r.split == "calibration"]
        require(
            len(fresh) == 2644
            and len({r.group_id for r in fresh}) == 1084
            and not {r.group_id for r in fresh} & {r.group_id for r in previous}
            and not {normalized_state(r) for r in fresh if not r.request.state.images}
            & {normalized_state(r) for r in previous if not r.request.state.images}
            and not {r.image_sha256 for r in fresh if r.image_sha256}
            & {r.image_sha256 for r in previous if r.image_sha256},
            "new validation observations overlap inspected or earlier validation populations",
        )
        previous.extend(fresh)
        datasets[role] = path
        for file in (
            Path(spec["design"]),
            path,
            path.parent / "protocol.json",
            path.parent / "independent-target-audit.json",
        ):
            evidence[str(file)] = digest(file)
        for file, expected in data_protocol["excluded_record_files"].items():
            require(digest(file) == expected, "excluded population changed")
            evidence[file] = expected
    for path in (
        DESIGN_PATH,
        REPAIR_PATH,
        design_path,
        fit_path,
        fit_path.parent / "protocol.json",
    ):
        evidence[str(path)] = digest(path)
    sources["scripts/cohort_policy_checks.py"] = digest(__file__)
    evidence.update(sources)
    return {
        "config": config,
        "original_selection": original,
        "datasets": datasets,
        "cohort_groups": cohort_groups,
        "depth_details": prior["depth_details"],
        "evidence_files": evidence,
        "source_files": sources,
        "data_separation": {
            "fitting_questions": 7932,
            "fitting_groups": 3252,
            "validation_groups_each": 1084,
            "validation_populations": 2,
            "group_overlap": 0,
            "normalized_state_overlap": 0,
            "image_byte_overlap": 0,
            "original_noncalibration_splits_identical": True,
            "previously_failed_validation_reused_as_fitting": True,
        },
    }


def expected_selection(design):
    return {
        **design["original_selection"],
        "records_sha256": digest(design["datasets"]["fit"]),
        "study_protocol_sha256": digest(DESIGN_PATH),
        "calibration_design_sha256": design["config"]["fitting_data_design_sha256"],
        "release_extensions": {
            "workflow_cohort_policy_v12": {
                "study_protocol_sha256": digest(DESIGN_PATH),
                "calibration_design_sha256": design["config"]["fitting_data_design_sha256"],
                "validation_design_sha256": {
                    key: value["design_sha256"]
                    for key, value in design["config"]["validation_populations"].items()
                },
            }
        },
        "original_development_selection_sha256": design["config"]["development_selection_sha256"],
        "candidate_weights_reselected": False,
        "calibration_method_declared_after_inspected_failure": True,
        "independent_policy_validation_records": {
            key: {"path": str(value), "sha256": digest(value)}
            for key, value in design["datasets"].items()
            if key != "fit"
        },
    }


def collect_cohort_study(selection_path, records):
    result = collect_cohort_design()
    selection = read(selection_path)
    require(selection == expected_selection(result), "cohort continuation selection changed")
    require(Path(records) == result["datasets"]["fit"], "wrong combined corpus")
    result["release_extensions"] = selection["release_extensions"]
    result["evidence_files"][str(selection_path)] = digest(selection_path)
    return result


def collect_cohort_evidence(checkpoint, selection_path, records, calibration_report, policy_report):
    result = collect_cohort_study(selection_path, records)
    config = result["config"]
    root, checkpoint = Path(policy_report).parent, Path(checkpoint)
    manifest, calibration = read(checkpoint / "manifest.json"), read(calibration_report)
    selection, freeze = read(selection_path), read(root / "policy-freeze.json")
    weights, manifest_hash = (
        digest(checkpoint / "head.safetensors"),
        digest(checkpoint / "manifest.json"),
    )
    require(
        weights == selection["weights_sha256"] == manifest["weights_sha256"]
        and manifest["training"]["release_extensions"] == result["release_extensions"]
        and calibration["calibrated_manifest_sha256"] == manifest_hash
        and calibration["calibration"] == manifest["calibration"]
        and calibration["protocol"]["source_sha256"] == digest("scripts/fit_cohort_policy.py")
        and calibration["protocol"]["policy_design_sha256"] == digest(DESIGN_PATH)
        and calibration["protocol"]["selection_sha256"] == digest(selection_path)
        and calibration["protocol"]["records_sha256"] == digest(records)
        and calibration["protocol"]["weights_sha256"] == weights
        and freeze["study_sha256"] == digest(DESIGN_PATH)
        and freeze["selection_sha256"] == digest(selection_path)
        and freeze["weights_sha256"] == weights
        and freeze["manifest_sha256"] == manifest_hash
        and freeze["calibration_report_sha256"] == digest(calibration_report)
        and freeze["validation_records"] == selection["independent_policy_validation_records"]
        and freeze["independent_validation_predictions_observed"] is False
        and freeze["source_sha256"] == digest("scripts/freeze_cohort_policy.py"),
        "cohort calibrated policy identity changed",
    )
    require(
        set(calibration["fitting"]) == set(QUESTION_TYPES)
        and set(calibration["serialized_policy_check"]) == set(QUESTION_TYPES)
        and all(
            value["matches_fitted_policy"] is True
            for value in calibration["serialized_policy_check"].values()
        ),
        "serialized fitting policy failed",
    )
    for fit in calibration["fitting"].values():
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
            "a fitting cohort failed",
        )
    for path, expected in freeze["source_files"].items():
        require(digest(path) == expected, "cohort frozen source changed: " + path)
    old_path = config["previous_group_regression"]["population_report"]
    require(
        digest(old_path) == config["previous_group_regression"]["population_report_sha256"],
        "original policy groups changed",
    )
    old = read(old_path)
    old_groups = {key: set(value["policy_group_ids"]) for key, value in old["fitting"].items()}
    populations = [
        (role, result["datasets"][role], None, root / role, root / (role + "-policy.json"))
        for role in config["validation_populations"]
    ]
    populations.append(
        (
            "previous",
            Path(read(config["origin_study"])["records"]),
            old_groups,
            root / "previous-calibration",
            Path(policy_report),
        )
    )
    details, evidence, sources = {}, result["evidence_files"], result["source_files"]
    for role, population, groups, evaluation_root, report_path in populations:
        report, evaluation = read(report_path), read(evaluation_root / "evaluation.json")
        current, _ = policy_population(
            evaluation_root, population, manifest, selection_path, groups
        )
        require(
            report["passed"] is True
            and all(value["passed"] for value in current.values())
            and report["by_type"] == current
            and report["kind"] == role
            and report["weights_sha256"] == weights
            and report["manifest_sha256"]
            == evaluation["protocol"]["manifest_sha256"]
            == manifest_hash
            and report["policy_freeze_sha256"] == digest(root / "policy-freeze.json")
            and report["evaluation_sha256"] == digest(evaluation_root / "evaluation.json")
            and report["predictions_sha256"] == digest(evaluation_root / "predictions.jsonl")
            and report["selection_sha256"] == digest(selection_path)
            and report["study_sha256"] == digest(DESIGN_PATH)
            and report["fresh_calibration_report_sha256"] == digest(calibration_report)
            and report["source_sha256"] == digest("scripts/audit_cohort_policy_population.py")
            and report["policy_refitted"] is False
            and report["population_previously_inspected"] is (role == "previous")
            and report["final_predictions_used"] is False
            and datetime.fromisoformat(evaluation["protocol"]["started_at_utc"])
            >= datetime.fromisoformat(freeze["frozen_at_utc"]),
            "independent cohort policy evaluation failed or changed",
        )
        if groups is not None:
            require(
                all(
                    current[key]["questions"] == value["questions"]
                    for key, value in old["fitting"].items()
                ),
                "original group counts changed",
            )
        details[role] = current
        for path in (
            report_path,
            evaluation_root / "evaluation.json",
            evaluation_root / "predictions.jsonl",
        ):
            evidence[str(path)] = digest(path)
    for path in (
        checkpoint / "manifest.json",
        checkpoint / "head.safetensors",
        Path(calibration_report),
        root / "policy-freeze.json",
        Path(old_path),
    ):
        evidence[str(path)] = digest(path)
    sources.update(freeze["source_files"])
    for path in (
        "scripts/run_cohort_policy_release.py",
        "scripts/prepare_cohort_policy_runtime.py",
    ):
        sources[path] = digest(path)
    evidence.update(sources)
    return {
        "release_extensions": result["release_extensions"],
        "checks": {
            "depth_study_provenance": True,
            "cohort_policy_study_provenance": True,
            "independent_policy_validation": True,
            "previous_calibration_policy_regression": True,
        },
        "details": {
            "depth_study_provenance": result["depth_details"],
            "cohort_policy_study_provenance": {
                "data_separation": result["data_separation"],
                "per_cohort_fit_error_max": 0.12,
                "candidate_weights_reselected": False,
            },
            "independent_policy_validation": {
                key: value for key, value in details.items() if key != "previous"
            },
            "previous_calibration_policy_regression": details["previous"],
        },
        "evidence_files": evidence,
        "source_files": sources,
        "previous_policy_report": str(policy_report),
        "previous_policy_report_sha256": digest(policy_report),
    }
