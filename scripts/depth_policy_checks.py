"""Bind conservative policy calibration to the unchanged depth model and fresh data."""

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from depth_release_checks import collect_depth_study

from veyra.constants import QUESTION_TYPES
from veyra.data import read_records

DESIGN_PATH = Path("configs/workflow-depth-policy-v12.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Conservative policy rejected: " + message)


def normalized_state(record):
    return " ".join(
        re.sub(r"Case reference [0-9a-f]+", "", record.request.state.text).lower().split()
    )


def collect_policy_design():
    config = read(DESIGN_PATH)
    require(config["protocol"] == "workflow-depth-conservative-policy-v12", "unknown design")
    origin = read(config["origin_study"])
    origin_design = read(origin["fresh_calibration_design"])
    original = read(config["development_selection"])
    result = collect_depth_study(
        config["development_selection"], Path(origin_design["output"]) / "records.jsonl"
    )
    evidence, sources = result["evidence_files"], result["source_files"]
    evidence[str(DESIGN_PATH)] = digest(DESIGN_PATH)
    require(
        digest(config["origin_study"]) == config["origin_study_sha256"]
        and digest(config["development_selection"]) == config["development_selection_sha256"]
        and original["eligible"] is True
        and original["weights_sha256"] == config["weights_sha256"]
        and original["manifest_sha256"] == config["uncalibrated_manifest_sha256"]
        and config["release_protocol_sha256"] == original["release_protocol_sha256"]
        and config["required_priorities"] == [1, 2],
        "original weight selection changed",
    )
    require(
        config["method"]["policy_fit_expected_error_max"] == 0.12
        and config["method"]["policy_fit_minimum_coverage"] == 0.6
        and config["method"]["one_policy_only"] is True
        and config["method"]["candidate_weights_reselected"] is False,
        "predeclared policy method changed",
    )
    for key in ("source_files", "frozen_generator_ancestors"):
        for path, expected in config[key].items():
            require(digest(path) == expected, "preparation source changed: " + path)
            sources[path] = expected
    trigger = config["trigger"]
    for key in ("previous_policy_report", "diagnosis"):
        require(digest(trigger[key]) == trigger[key + "_sha256"], "failure evidence changed")
        evidence[trigger[key]] = digest(trigger[key])
    require(read(trigger["previous_policy_report"])["passed"] is False, "failure was not retained")
    old_records = read_records(Path(origin["records"]))
    prior_records = read_records(Path(origin_design["output"]) / "records.jsonl")
    old_groups = {r.group_id for r in old_records + prior_records}
    old_states = {
        normalized_state(r) for r in old_records + prior_records if not r.request.state.images
    }
    calibration_groups, calibration_states, calibration_images = {}, {}, {}
    datasets = {}
    for role, key in (("fit", "fitting_data_design"), ("validation", "validation_data_design")):
        design_path = Path(config[key])
        require(digest(design_path) == config[key + "_sha256"], "data design changed")
        design = read(design_path)
        root = Path(design["output"])
        path = root / "records.jsonl"
        protocol, audit = read(root / "protocol.json"), read(root / "independent-target-audit.json")
        require(
            protocol["records_sha256"] == audit["records_sha256"] == digest(path)
            and protocol["calibration_design_sha256"]
            == audit["calibration_design_sha256"]
            == digest(design_path)
            and protocol["study_protocol_sha256"] == digest(DESIGN_PATH)
            and protocol["model_outputs_used"] is False
            and protocol["generator_sha256"] == sources["scripts/build_followup_calibration.py"]
            and audit["source_sha256"] == sources["scripts/verify_followup_calibration_data.py"]
            and audit["passed"] is True
            and audit["fresh_policy_targets_recomputed"] == 2040
            and audit["maximum_target_absolute_error"] <= 1e-7
            and datetime.fromisoformat(protocol["created_at_utc"])
            > datetime.fromisoformat(config["declared_at_utc"]),
            "fresh data lacks a prospective independent target audit",
        )
        records = read_records(path)
        for split in ("train", "dev", "test"):
            before = [r.model_dump_json() for r in old_records if r.split == split]
            after = [r.model_dump_json() for r in records if r.split == split]
            require(before == after, "non-calibration data changed")
        fresh = [r for r in records if r.split == "calibration"]
        groups = {r.group_id for r in fresh}
        states = {normalized_state(r) for r in fresh if not r.request.state.images}
        images = {r.image_sha256 for r in fresh if r.image_sha256}
        require(
            len(fresh) == 2644
            and len(groups) == 1084
            and not groups & old_groups
            and not states & old_states,
            "fresh population size or separation changed",
        )
        calibration_groups[role], calibration_states[role], calibration_images[role] = (
            groups,
            states,
            images,
        )
        datasets[role] = path
        for file in (
            design_path,
            path,
            root / "protocol.json",
            root / "independent-target-audit.json",
        ):
            evidence[str(file)] = digest(file)
        for file, expected in protocol["excluded_record_files"].items():
            require(digest(file) == expected, "data exclusion population changed")
            evidence[file] = expected
    require(
        not calibration_groups["fit"] & calibration_groups["validation"]
        and not calibration_states["fit"] & calibration_states["validation"]
        and not calibration_images["fit"] & calibration_images["validation"],
        "fitting and independent validation overlap",
    )
    sources["scripts/depth_policy_checks.py"] = digest(__file__)
    evidence.update(sources)
    return {
        "config": config,
        "original_selection": original,
        "datasets": datasets,
        "evidence_files": evidence,
        "source_files": sources,
        "depth_details": result["details"],
        "data_separation": {
            "fit_groups": len(calibration_groups["fit"]),
            "validation_groups": len(calibration_groups["validation"]),
            "cross_population_group_overlap": 0,
            "cross_population_normalized_state_overlap": 0,
            "cross_population_image_byte_overlap": 0,
            "original_noncalibration_splits_identical": True,
        },
    }


def expected_selection(design):
    config, original = design["config"], design["original_selection"]
    return {
        **original,
        "records_sha256": digest(design["datasets"]["fit"]),
        "study_protocol_sha256": digest(DESIGN_PATH),
        "calibration_design_sha256": config["fitting_data_design_sha256"],
        "release_extensions": {
            "workflow_depth_policy_v12": {
                "study_protocol_sha256": digest(DESIGN_PATH),
                "calibration_design_sha256": config["fitting_data_design_sha256"],
                "validation_design_sha256": config["validation_data_design_sha256"],
            }
        },
        "original_development_selection": config["development_selection"],
        "original_development_selection_sha256": config["development_selection_sha256"],
        "candidate_weights_reselected": False,
        "calibration_method_declared_after_inspected_failure": True,
        "independent_policy_validation_records": str(design["datasets"]["validation"]),
        "independent_policy_validation_records_sha256": digest(design["datasets"]["validation"]),
    }


def collect_policy_study(selection_path, records_path):
    result = collect_policy_design()
    selection = read(selection_path)
    require(selection == expected_selection(result), "continuation weight selection changed")
    require(Path(records_path) == result["datasets"]["fit"], "wrong fitting/final corpus")
    result["evidence_files"][str(selection_path)] = digest(selection_path)
    result["release_extensions"] = selection["release_extensions"]
    return result


def policy_population(evaluation_root, records_path, manifest, selection_path, groups_by_type=None):
    """Reconstruct an unchanged policy's risk from its complete saved prediction population."""
    root = Path(evaluation_root)
    evaluation = read(root / "evaluation.json")
    predictions = root / "predictions.jsonl"
    require(
        evaluation["protocol"]["records_sha256"] == digest(records_path)
        and evaluation["protocol"]["selection_sha256"] == digest(selection_path)
        and evaluation["protocol"]["weights_sha256"] == manifest["weights_sha256"]
        and evaluation["protocol"]["source_sha256"] == digest("scripts/evaluate_workflow.py")
        and evaluation["protocol"]["arguments"]["split"] == "calibration"
        and evaluation["predictions_sha256"] == digest(predictions)
        and evaluation["calibration"] == manifest["calibration"]
        and evaluation["fitting"] is None
        and evaluation["calibrated_manifest_sha256"] is None,
        "validation outputs changed or refitted the policy",
    )
    records = {r.id: r for r in read_records(Path(records_path)) if r.split == "calibration"}
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    require(
        len({r["id"] for r in rows}) == len(rows) and {r["id"] for r in rows} == set(records),
        "validation population differs from its declared records",
    )
    calibration = manifest["calibration"]
    for row in rows:
        record = records[row["id"]]
        name = row["question"]
        require(
            name in record.request.questions
            and row["group"] == record.group_id
            and row["type"] == record.request.questions[name].type,
            "validation identity changed",
        )
        require(
            row["prediction"] in record.targets[name]
            and isinstance(row["confidence"], (int, float))
            and 0 <= row["confidence"] <= 1,
            "invalid prediction or confidence",
        )
        gold = next((tag[9:] for tag in record.tags if tag.startswith("gold_key:")), None)
        error = (
            float(row["prediction"] != gold)
            if "teacher_distribution" in record.tags and gold is not None
            else 1 - record.targets[name][row["prediction"]]
        )
        type_id = QUESTION_TYPES.index(row["type"])
        abstained = (
            row["type"] in calibration["always_abstain_types"]
            or row["confidence"] < calibration["abstain_thresholds"][type_id]
        )
        require(
            abs(error - row["expected_error"]) <= 1e-9 and row["abstained"] is abstained,
            "reported error or abstention differs from the serialized policy",
        )
    results, selected_rows = {}, {}
    for kind in QUESTION_TYPES:
        selected = [
            r
            for r in rows
            if r["type"] == kind and (groups_by_type is None or r["group"] in groups_by_type[kind])
        ]
        if groups_by_type is not None:
            require(
                {r["group"] for r in selected} == groups_by_type[kind], "regression groups changed"
            )
        require(bool(selected), "empty validation type")
        accepted = [r for r in selected if not r["abstained"]]
        coverage = len(accepted) / len(selected)
        error = sum(r["expected_error"] for r in accepted) / len(accepted) if accepted else None
        results[kind] = {
            "questions": len(selected),
            "accepted": len(accepted),
            "coverage": coverage,
            "expected_error": error,
            "passed": coverage >= 0.6 and error is not None and error <= 0.15,
        }
        selected_rows[kind] = selected
    return results, selected_rows


def bootstrap_policy(rows, seed=223, replicates=2000):
    """Paired group resampling of the frozen policy and always-answering predictions."""
    groups = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    for row in rows:
        accepted = not row["abstained"]
        groups[row["group"]] += [
            1,
            row["expected_error"],
            accepted,
            accepted * row["expected_error"],
        ]
    values = np.stack([groups[key] for key in sorted(groups)])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        count, loss, answers, accepted_loss = values[
            rng.integers(len(values), size=len(values))
        ].sum(0)
        if answers:
            samples.append(
                [answers / count, accepted_loss / answers, accepted_loss / answers - loss / count]
            )
    require(len(samples) == replicates, "bootstrap sampled no accepted observations")
    intervals = np.quantile(np.asarray(samples), [0.025, 0.975], axis=0)
    return {
        "groups": len(groups),
        "seed": seed,
        "replicates": replicates,
        "coverage_95_interval": intervals[:, 0].tolist(),
        "expected_error_95_interval": intervals[:, 1].tolist(),
        "error_delta_policy_minus_always_answer_95_interval": intervals[:, 2].tolist(),
        "used_for_policy_fitting_or_selection": False,
        "limitation": (
            "Descriptive observation-group bootstrap; "
            "not a distribution-free risk guarantee or a final result."
        ),
    }


def collect_policy_evidence(checkpoint, selection_path, records, calibration_report, policy_report):
    result = collect_policy_study(selection_path, records)
    config = result["config"]
    checkpoint = Path(checkpoint)
    root = Path(policy_report).parent
    selection, calibration, manifest = (
        read(selection_path),
        read(calibration_report),
        read(checkpoint / "manifest.json"),
    )
    freeze = read(root / "policy-freeze.json")
    for path, expected in freeze["source_files"].items():
        require(digest(path) == expected, "frozen policy source changed: " + path)
    validation = read(root / "independent-policy-validation.json")
    previous = read(policy_report)
    weights, manifest_hash = (
        digest(checkpoint / "head.safetensors"),
        digest(checkpoint / "manifest.json"),
    )
    require(
        weights == selection["weights_sha256"] == manifest["weights_sha256"]
        and manifest["training"]["release_extensions"] == result["release_extensions"]
        and calibration["calibrated_manifest_sha256"] == manifest_hash
        and calibration["protocol"]["weights_sha256"] == weights
        and calibration["protocol"]["manifest_sha256"] == selection["manifest_sha256"]
        and calibration["protocol"]["selection_sha256"] == digest(selection_path)
        and calibration["protocol"]["policy_design_sha256"] == digest(DESIGN_PATH)
        and calibration["protocol"]["records_sha256"] == digest(records)
        and calibration["protocol"]["source_sha256"] == digest("scripts/evaluate_workflow.py")
        and freeze["weights_sha256"] == weights
        and freeze["manifest_sha256"] == manifest_hash
        and freeze["calibration_report_sha256"] == digest(calibration_report)
        and freeze["selection_sha256"] == digest(selection_path)
        and freeze["study_sha256"] == digest(DESIGN_PATH)
        and freeze["validation_records_sha256"] == digest(result["datasets"]["validation"])
        and freeze["source_sha256"] == digest("scripts/freeze_depth_policy.py")
        and freeze["independent_validation_predictions_observed"] is False
        and calibration["protocol"]["arguments"]["split"] == "calibration"
        and calibration["calibration"] == manifest["calibration"],
        "frozen calibrated model identity changed",
    )
    require(
        set(calibration["fitting"]) == set(QUESTION_TYPES)
        and all(
            fit["passed"] is True
            and fit["always_abstain"] is False
            and fit["policy_fit_expected_error_max"] == 0.12
            and fit["expected_error"] <= 0.12
            and fit["coverage"] >= 0.6
            for fit in calibration["fitting"].values()
        )
        and set(calibration["serialized_policy_check"]) == set(QUESTION_TYPES)
        and all(
            v["matches_fitted_policy"] is True
            for v in calibration["serialized_policy_check"].values()
        ),
        "conservative calibration fit failed",
    )
    original_report_path = config["previous_group_regression"]["population_report"]
    require(
        digest(original_report_path)
        == config["previous_group_regression"]["population_report_sha256"],
        "original regression group selection changed",
    )
    original_report = read(original_report_path)
    original_groups = {k: set(v["policy_group_ids"]) for k, v in original_report["fitting"].items()}
    old_records = read(config["origin_study"])["records"]
    evidence, sources = result["evidence_files"], result["source_files"]
    population_results = {}
    for name, evaluation_root, population, groups, report in (
        (
            "independent_policy_validation",
            root / "policy-validation",
            result["datasets"]["validation"],
            None,
            validation,
        ),
        (
            "previous_calibration_policy_regression",
            root / "previous-calibration",
            old_records,
            original_groups,
            previous,
        ),
    ):
        evaluation = read(evaluation_root / "evaluation.json")
        details, _ = policy_population(
            evaluation_root, population, manifest, selection_path, groups
        )
        require(
            report["passed"] is True
            and all(v["passed"] for v in details.values())
            and report["by_type"] == details
            and report["manifest_sha256"]
            == evaluation["protocol"]["manifest_sha256"]
            == manifest_hash
            and report["weights_sha256"] == weights
            and report["evaluation_sha256"] == digest(evaluation_root / "evaluation.json")
            and report["predictions_sha256"] == digest(evaluation_root / "predictions.jsonl")
            and report["policy_freeze_sha256"] == digest(root / "policy-freeze.json")
            and report["policy_refitted"] is False
            and report["study_sha256"] == digest(DESIGN_PATH)
            and report["selection_sha256"] == digest(selection_path)
            and report["fresh_calibration_report_sha256"] == digest(calibration_report)
            and report["kind"] == ("previous" if groups is not None else "validation")
            and report["population_previously_inspected"] is (groups is not None)
            and report["final_predictions_used"] is False
            and report["source_sha256"] == digest("scripts/audit_depth_policy_population.py")
            and datetime.fromisoformat(evaluation["protocol"]["started_at_utc"])
            >= datetime.fromisoformat(freeze["frozen_at_utc"]),
            "independent or previous-group policy requirement failed",
        )
        if groups is not None:
            require(
                all(
                    details[k]["questions"] == v["questions"]
                    for k, v in original_report["fitting"].items()
                ),
                "old group counts changed",
            )
        population_results[name] = details
        for path in (evaluation_root / "evaluation.json", evaluation_root / "predictions.jsonl"):
            evidence[str(path)] = digest(path)
    for path in (
        checkpoint / "manifest.json",
        checkpoint / "head.safetensors",
        Path(calibration_report),
        Path(policy_report),
        root / "independent-policy-validation.json",
        root / "policy-freeze.json",
    ):
        evidence[str(path)] = digest(path)
    evidence[original_report_path] = digest(original_report_path)
    sources.update(freeze["source_files"])
    for path in (
        "scripts/audit_depth_policy_population.py",
        "scripts/freeze_depth_policy.py",
        "scripts/evaluate_workflow.py",
        "scripts/run_depth_policy_release.py",
    ):
        sources[path] = digest(path)
    evidence.update(sources)
    return {
        "release_extensions": result["release_extensions"],
        "checks": {
            "depth_study_provenance": True,
            "conservative_policy_study_provenance": True,
            "independent_policy_validation": True,
            "previous_calibration_policy_regression": True,
        },
        "details": {
            "depth_study_provenance": result["depth_details"],
            "conservative_policy_study_provenance": {
                "data_separation": result["data_separation"],
                "fit_error_target": 0.12,
                "release_error_limit": 0.15,
                "no_candidate_reselection": True,
            },
            **population_results,
        },
        "evidence_files": evidence,
        "source_files": sources,
        "previous_policy_report": str(policy_report),
        "previous_policy_report_sha256": digest(policy_report),
    }
