"""Bind the continuation study and its extra calibration gate before final inference."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from veyra.constants import QUESTION_TYPES
from veyra.data import read_records


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_workspace_evidence(
    checkpoint, selection_path, records, calibration_report, policy_report
):
    evidence, sources = {}, {}

    def read(path):
        path = Path(path)
        evidence[str(path)] = digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def require(condition, message):
        if not condition:
            raise ValueError("Continuation evidence rejected: " + message)

    config_path = Path("configs/workflow-workspace-study-v12.json")
    design_path = Path("configs/workflow-workspace-calibration-v12.json")
    config, design = read(config_path), read(design_path)
    selection, calibration = read(selection_path), read(calibration_report)
    policy = read(policy_report)
    checkpoint, records = Path(checkpoint), Path(records)
    manifest = read(checkpoint / "manifest.json")
    weights, manifest_hash = (
        digest(checkpoint / "head.safetensors"),
        digest(checkpoint / "manifest.json"),
    )
    expected_extension = {
        "workflow_workspace_v12": {
            "study_protocol_sha256": digest(config_path),
            "calibration_design_sha256": digest(design_path),
        }
    }
    require(
        selection["release_extensions"]
        == manifest["training"]["release_extensions"]
        == expected_extension
        and selection["eligible"] is True
        and selection["final_or_calibration_used"] is False
        and selection["study_protocol_sha256"] == digest(config_path)
        and selection["calibration_design_sha256"] == digest(design_path)
        and selection["weights_sha256"] == policy["weights_sha256"] == weights
        and policy["manifest_sha256"] == manifest_hash
        and policy["selection_sha256"] == digest(selection_path)
        and policy["study_sha256"] == digest(config_path)
        and policy["design_sha256"] == digest(design_path)
        and policy["fresh_calibration_report_sha256"] == digest(calibration_report)
        and policy["policy_refitted"] is False,
        "study, selected model, fresh policy or policy regression changed",
    )
    require(
        digest(config["release_protocol"]) == config["release_protocol_sha256"]
        and digest(config["records"]) == config["records_sha256"] == design["old_records_sha256"]
        and digest(Path(config["parent"]) / "head.safetensors") == config["parent_weights_sha256"]
        and digest(Path(config["parent"]) / "manifest.json") == config["parent_manifest_sha256"],
        "original experiment or parent changed",
    )
    protocol = read(records.parent / "protocol.json")
    audit = read(records.parent / "independent-target-audit.json")
    require(
        digest(records)
        == protocol["records_sha256"]
        == audit["records_sha256"]
        == selection["records_sha256"]
        and protocol["calibration_design_sha256"] == digest(design_path)
        and protocol["study_protocol_sha256"] == digest(config_path)
        and audit["passed"] is True
        and audit["fresh_policy_targets_recomputed"] == 2040
        and audit["maximum_target_absolute_error"] <= 1e-12
        and protocol["model_outputs_used"] is False
        and protocol["old_calibration_policy_regression_required"] is True
        and protocol["known_workflow_labels_are_new_explicit_policy_targets"] is True,
        "fresh calibration design or independently computed targets changed",
    )
    original = read_records(Path(config["records"]))
    fresh = read_records(records)
    for split in ("train", "dev", "test"):
        before = "".join(r.model_dump_json() + "\n" for r in original if r.split == split).encode()
        after = "".join(r.model_dump_json() + "\n" for r in fresh if r.split == split).encode()
        require(
            before == after
            and hashlib.sha256(after).hexdigest()
            == audit["preserved_subset_sha256"][split]
            == protocol["preserved_subset_sha256"][split],
            "frozen non-calibration records changed: " + split,
        )
    require(
        not {r.group_id for r in original}
        & {r.group_id for r in fresh if r.split == "calibration"},
        "fresh calibration reused a previous group",
    )
    selection_protocol = read(Path(selection_path).parent / "protocol.json")
    require(
        selection_protocol["data_audit_sha256"]
        == digest(records.parent / "independent-target-audit.json"),
        "data audit changed since selection",
    )
    require(
        selection_protocol["study_protocol_sha256"] == digest(config_path)
        and selection_protocol["calibration_design_sha256"] == digest(design_path)
        and datetime.fromisoformat(protocol["created_at_utc"])
        <= datetime.fromisoformat(selection_protocol["started_at_utc"]),
        "fresh calibration was not fixed before candidate selection",
    )
    expected_candidates = [Path(config["parent"])]
    for arm in config["execution_order"]:
        run = Path("runs/workflow-v12-workspace-study") / arm
        training = read(run / "protocol.json")
        complete = read(run / "complete.json")
        read(run / "history.json")
        require(
            training["config_sha256"] == digest(config_path)
            and training["arm"] == arm
            and training["calibration_or_final_used"] is False
            and complete["completed"] is True
            and datetime.fromisoformat(config["declared_at_utc"])
            <= datetime.fromisoformat(training["started_at_utc"]),
            "training was not the declared separate study",
        )
        candidates = [run / f"epoch-{i}" / "checkpoint" for i in range(1, config["epochs"] + 1)]
        require(list(map(Path, complete["candidates"])) == candidates, "undeclared candidate")
        for candidate in candidates:
            model_manifest = read(candidate / "manifest.json")
            require(
                model_manifest["training"]["protocol_sha256"] == digest(run / "protocol.json"),
                "candidate training provenance mismatch",
            )
        expected_candidates.extend(candidates)
        sources.update(training["source_sha256"])
    require(
        [Path(row["checkpoint"]) for row in selection["candidates"]] == expected_candidates,
        "development comparison population changed",
    )
    for row in selection["candidates"]:
        candidate = Path(row["checkpoint"])
        require(
            row["weights_sha256"] == digest(candidate / "head.safetensors")
            and row["manifest_sha256"] == digest(candidate / "manifest.json"),
            "a development candidate changed",
        )
    baseline_development = read(selection["baseline_merged_dev"])
    for row in [baseline_development, *selection["candidates"]]:
        path = row["development_logits"]
        require(digest(path) == row["development_logits_sha256"], "development logits changed")
        evidence[path] = row["development_logits_sha256"]
    require(
        all(
            row["maximum_coverage_at_15pct_error"] >= 0.6
            for row in selection["selected"]["development_risk_diagnostic"].values()
        )
        and set(selection["selected"]["development_risk_diagnostic"]) == set(QUESTION_TYPES),
        "development abstention screen failed",
    )
    gradient = read("runs/workflow-v12-workspace-study/primitive/auxiliary-gradient-path.json")
    require(
        gradient["same_decision_forward"] is True
        and gradient["backbone_parameters_with_nonzero_gradient"] > 0,
        "primitive loss did not reach the backbone",
    )
    old_path = Path("runs/workflow-v12-evaluation/calibration/evaluation.json")
    old = read(old_path)
    regression_root = Path(policy_report).parent / "previous-calibration"
    regression = read(regression_root / "evaluation.json")
    prediction_path = regression_root / "predictions.jsonl"
    evidence[str(prediction_path)] = digest(prediction_path)
    require(
        digest(old_path)
        == config["trigger"]["evaluation_sha256"]
        == policy["original_calibration_report_sha256"]
        and digest(regression_root / "evaluation.json") == policy["evaluation_sha256"]
        and digest(prediction_path)
        == policy["predictions_sha256"]
        == regression["predictions_sha256"]
        and regression["protocol"]["weights_sha256"] == weights
        and regression["protocol"]["manifest_sha256"] == manifest_hash
        and regression["protocol"]["records_sha256"] == config["records_sha256"]
        and regression["protocol"]["selection_sha256"] == digest(selection_path)
        and regression["protocol"]["source_sha256"] == digest("scripts/evaluate_workflow.py")
        and regression["fitting"] is None
        and regression["calibration"] == calibration["calibration"] == manifest["calibration"],
        "previous calibration was changed or used for refitting",
    )
    predictions = [
        json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    require(
        len({row["id"] for row in predictions}) == len(predictions), "duplicate prediction rows"
    )
    gate, details = design["additional_mandatory_gate"], {}
    for kind in QUESTION_TYPES:
        groups = set(old["fitting"][kind]["policy_group_ids"])
        rows = [r for r in predictions if r["type"] == kind and r["group"] in groups]
        require(
            len(rows) == old["fitting"][kind]["questions"] and {r["group"] for r in rows} == groups,
            "old policy population changed",
        )
        accepted = [r for r in rows if not r["abstained"]]
        coverage = len(accepted) / len(rows)
        error = sum(r["expected_error"] for r in accepted) / len(accepted) if accepted else None
        details[kind] = {
            "questions": len(rows),
            "accepted": len(accepted),
            "coverage": coverage,
            "expected_error": error,
            "passed": coverage >= gate["per_type_minimum_coverage"]
            and error is not None
            and error <= gate["per_type_expected_error_max"],
        }
    require(
        details == policy["by_type"]
        and policy["passed"] is True
        and all(r["passed"] for r in details.values()),
        "unchanged-policy regression failed",
    )
    for filename in (
        "scripts/select_workspace_checkpoint.py",
        "scripts/audit_previous_calibration_policy.py",
        "scripts/workspace_release_checks.py",
        "scripts/run_workspace_evaluation.py",
        "scripts/prepare_workspace_calibration_sources.py",
        "scripts/build_workspace_calibration.py",
        "scripts/verify_workspace_calibration_data.py",
    ):
        sources[filename] = digest(filename)
    sources.update(selection_protocol["source_files"])
    require(
        protocol["generator_sha256"] == sources["scripts/build_workspace_calibration.py"]
        and audit["source_sha256"] == sources["scripts/verify_workspace_calibration_data.py"]
        and policy["source_sha256"] == sources["scripts/audit_previous_calibration_policy.py"],
        "data builder, independent verifier or policy auditor changed",
    )
    for path, expected in sources.items():
        require(digest(path) == expected, "source changed: " + path)
    evidence.update(sources)
    return {
        "release_extensions": expected_extension,
        "checks": {
            "workspace_study_provenance": True,
            "previous_calibration_policy_regression": True,
        },
        "details": {
            "previous_calibration_policy_regression": details,
            "workspace_study_provenance": {
                "one_training_seed": True,
                "calibration_labels_changed": True,
                "original_noncalibration_splits_preserved": True,
            },
        },
        "evidence_files": evidence,
        "source_files": sources,
        "previous_policy_report": str(policy_report),
        "previous_policy_report_sha256": digest(policy_report),
    }
