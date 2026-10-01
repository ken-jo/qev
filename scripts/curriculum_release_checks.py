"""Recompute selection and bind the curriculum study and unchanged calibration regression."""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_workflow_workspace import development_risk

from veyra.constants import QUESTION_TYPES
from veyra.data import TrainingRecord, read_records
from veyra.workflow_learning import annotate, selection_key, summarize_logits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Curriculum evidence rejected: " + message)


def read_plain(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def same_numbers(actual, reported, where="metrics"):
    if isinstance(actual, dict):
        require(isinstance(reported, dict) and set(actual) == set(reported), where + " keys")
        for key, value in actual.items():
            same_numbers(value, reported[key], where + "/" + str(key))
    elif isinstance(actual, (list, tuple)):
        require(isinstance(reported, (list, tuple)) and len(actual) == len(reported), where)
        for i, (left, right) in enumerate(zip(actual, reported, strict=True)):
            same_numbers(left, right, where + "/" + str(i))
    elif isinstance(actual, float):
        require(
            isinstance(reported, (int, float))
            and math.isfinite(actual)
            and math.isfinite(reported)
            and abs(actual - reported) <= 1e-6,
            where + " differs from raw predictions",
        )
    else:
        require(actual == reported, where + " differs from raw predictions")


def recompute_development(report, tensors, rows, indices):
    """Recalculate original metrics and confidence screening from the saved logits."""
    checkpoint = Path(report["checkpoint"])
    require(
        digest(checkpoint / "head.safetensors") == report["weights_sha256"]
        and digest(checkpoint / "manifest.json") == report["manifest_sha256"]
        and digest(report["development_logits"]) == report["development_logits_sha256"],
        "model or saved development output changed",
    )
    artifact = load_file(report["development_logits"])
    require(torch.equal(artifact["cache_indices"], indices), "development row order changed")
    logits = artifact["logits"]
    require(logits.shape == (len(indices), 16), "development output shape changed")
    temperatures = read_plain(checkpoint / "manifest.json")["calibration"]["temperatures"]
    metrics = summarize_logits(logits, tensors, rows, indices, temperatures)
    risk = development_risk(logits, tensors, indices, temperatures)
    same_numbers(metrics, report["metrics"])
    if "development_risk_diagnostic" in report:
        same_numbers(risk, report["development_risk_diagnostic"], "risk")
    return metrics, risk


def collect_curriculum_study(selection_path, records_path):
    evidence, sources = {}, {}

    def read(path):
        evidence[str(path)] = digest(path)
        return read_plain(path)

    config_path = Path("configs/workflow-curriculum-study-v12.json")
    config = read(config_path)
    design_path = Path(config["fresh_calibration_design"])
    design = read(design_path)
    selection = read(selection_path)
    records_path = Path(records_path)
    expected_extension = {
        "workflow_curriculum_v12": {
            "study_protocol_sha256": digest(config_path),
            "calibration_design_sha256": digest(design_path),
        }
    }
    require(
        selection["eligible"] is True
        and selection["final_or_calibration_used"] is False
        and selection["release_extensions"] == expected_extension
        and selection["study_protocol_sha256"] == digest(config_path)
        and selection["calibration_design_sha256"] == digest(design_path)
        and selection["release_protocol_sha256"] == config["release_protocol_sha256"],
        "selection is not an eligible declared curriculum result",
    )
    paths = {
        Path(config["release_protocol"]): config["release_protocol_sha256"],
        Path(config["records"]): config["records_sha256"],
        Path(config["parent"]) / "head.safetensors": config["parent_weights_sha256"],
        Path(config["parent"]) / "manifest.json": config["parent_manifest_sha256"],
        Path(config["training_plan"]): config["training_plan_sha256"],
        Path(config["skills"]) / "audit.json": config["skills_audit_sha256"],
        Path(config["skills"]) / "independent-target-audit.json": config[
            "skills_independent_audit_sha256"
        ],
        design_path: config["fresh_calibration_design_sha256"],
    }
    for path, expected in paths.items():
        require(digest(path) == expected, "declared input changed: " + str(path))
    read(config["release_protocol"])
    read(config["training_plan"])
    skill_root = Path(config["skills"])
    skill_audit, skill_verification = (
        read(skill_root / "audit.json"),
        read(skill_root / "independent-target-audit.json"),
    )
    require(
        skill_audit["passed"] is True
        and skill_verification["passed"] is True
        and skill_verification["data_audit_sha256"] == digest(skill_root / "audit.json")
        and skill_verification["questions_checked"] == 16080
        and skill_verification["maximum_rendered_target_error"] <= 1e-12
        and skill_audit["groups_by_split"] == {"train": 960, "dev": 300}
        and skill_audit["calibration_or_final_used"] is False,
        "skill data integrity or split separation failed",
    )
    for name, expected in skill_audit["files_sha256"].items():
        require(digest(skill_root / name) == expected, "derived skill data changed")
    skill_protocol = read(skill_root / "protocol.json")
    for path, expected in skill_protocol["source_sha256"].items():
        require(digest(path) == expected, "skill construction input changed: " + path)
        if Path(path).suffix == ".py":
            sources[path] = expected
    sources.update(skill_verification["source_sha256"])
    prior_path = Path("runs/workflow-v12-workspace-selection/selection.json")
    prior = read(prior_path)
    require(
        digest(prior_path) == config["prior_selection_sha256"] and prior["eligible"] is False,
        "parent study changed",
    )
    origin = read(design["study_protocol"])
    require(
        prior["study_protocol_sha256"] == digest(design["study_protocol"])
        and prior["candidates"][2]["weights_sha256"] == config["parent_weights_sha256"]
        and origin["records_sha256"] == config["records_sha256"] == design["old_records_sha256"],
        "parent or calibration-design ancestry changed",
    )
    diagnostic = Path("runs/workflow-v12-skill-diagnostic/results.json")
    read(diagnostic)
    require(
        digest(diagnostic) == config["prerequisite_diagnostic_sha256"],
        "prerequisite diagnosis changed",
    )
    data_protocol = read(records_path.parent / "protocol.json")
    data_audit = read(records_path.parent / "independent-target-audit.json")
    require(
        digest(records_path)
        == selection["records_sha256"]
        == data_protocol["records_sha256"]
        == data_audit["records_sha256"]
        and data_protocol["calibration_design_sha256"] == digest(design_path)
        and data_protocol["study_protocol_sha256"] == digest(design["study_protocol"])
        and data_protocol["model_outputs_used"] is False
        and data_protocol["old_calibration_policy_regression_required"] is True
        and data_protocol["known_workflow_labels_are_new_explicit_policy_targets"] is True
        and data_audit["passed"] is True
        and data_audit["fresh_policy_targets_recomputed"] == 2040
        and data_audit["maximum_target_absolute_error"] <= 1e-12
        and datetime.fromisoformat(data_protocol["created_at_utc"])
        <= datetime.fromisoformat(config["declared_at_utc"]),
        "fresh calibration was not the declared pre-existing uninspected population",
    )
    require(
        not Path("runs/workflow-v12-workspace-evaluation/calibration/evaluation.json").exists(),
        "the earlier study already inspected this fresh calibration population",
    )
    original, fresh = read_records(Path(config["records"])), read_records(records_path)
    for split in ("train", "dev", "test"):
        before = "".join(r.model_dump_json() + "\n" for r in original if r.split == split).encode()
        after = "".join(r.model_dump_json() + "\n" for r in fresh if r.split == split).encode()
        require(
            before == after
            and hashlib.sha256(after).hexdigest()
            == data_audit["preserved_subset_sha256"][split]
            == data_protocol["preserved_subset_sha256"][split],
            "non-calibration records changed: " + split,
        )
    require(
        not {r.group_id for r in original}
        & {r.group_id for r in fresh if r.split == "calibration"},
        "calibration reused previous groups",
    )
    features = Path(config["features"])
    cache = read(features / "manifest.json")
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
    selection_protocol = read(Path(selection_path).parent / "protocol.json")
    require(
        selection_protocol["study_protocol_sha256"] == digest(config_path)
        and selection_protocol["data_audit_sha256"]
        == digest(records_path.parent / "independent-target-audit.json")
        and selection_protocol["feature_cache"] == cache
        and selection_protocol["calibration_or_final_used_for_selection"] is False,
        "selection population or provenance changed",
    )
    root = Path("runs/workflow-v12-curriculum-study")
    completed = read(root / "complete.json")
    require(
        completed["completed"] is True and completed["configuration_sha256"] == digest(config_path),
        "training is incomplete",
    )
    expected_candidates = [Path(config["parent"])]
    for arm in config["arms"]:
        run = root / arm
        training, completion = read(run / "protocol.json"), read(run / "complete.json")
        read(run / "history.json")
        require(
            training["configuration_sha256"] == digest(config_path)
            and training["arm"] == arm
            and training["calibration_or_final_used"] is False
            and training["feature_cache"] == cache
            and training["training_plan_sha256"] == config["training_plan_sha256"]
            and training["skills_audit_sha256"] == config["skills_audit_sha256"]
            and completion["completed"] is True
            and datetime.fromisoformat(config["declared_at_utc"])
            <= datetime.fromisoformat(training["started_at_utc"]),
            "training does not match the prospective declaration",
        )
        candidates = [run / f"epoch-{i}" / "checkpoint" for i in range(1, config["epochs"] + 1)]
        require(list(map(Path, completion["candidates"])) == candidates, "candidate pool changed")
        for epoch, candidate in enumerate(candidates, 1):
            manifest = read(candidate / "manifest.json")
            require(
                manifest["training"]["protocol_sha256"] == digest(run / "protocol.json")
                and manifest["training"]["epochs"] == epoch
                and manifest["training"]["optimizer_steps"]
                == epoch * math.ceil(15052 / config["accumulation"]),
                "candidate does not match its declared epoch",
            )
        sources.update(training["source_sha256"])
        expected_candidates.extend(candidates)
    require(
        [Path(row["checkpoint"]) for row in selection["candidates"]] == expected_candidates,
        "selection omitted or added a candidate",
    )
    baseline = read(selection["baseline_merged_dev"])
    require(
        digest(selection["baseline_merged_dev"])
        == selection["baseline_merged_dev_sha256"]
        == config["baseline_development_sha256"],
        "baseline development report changed",
    )
    rows, tensors = (
        read_plain(features / "records.json"),
        load_file(features / "features.safetensors"),
    )
    annotate(rows, [r for r in original if r.split in {"train", "dev"}])
    indices = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    torch.set_num_threads(4)
    baseline_metrics, _ = recompute_development(baseline, tensors, rows, indices)
    ranks = []
    for report in [baseline, *selection["candidates"]]:
        require(
            report["records_sha256"] == selection["records_sha256"],
            "development data identity changed",
        )
        evidence[report["development_logits"]] = report["development_logits_sha256"]
    for report in selection["candidates"]:
        metrics, risk = recompute_development(report, tensors, rows, indices)
        primary = selection_key(metrics, baseline_metrics)
        passes = set(risk) == set(QUESTION_TYPES) and all(
            r["maximum_coverage_at_15pct_error"] >= 0.6 for r in risk.values()
        )
        rank = (bool(primary[0] and passes), *primary)
        same_numbers(rank, report["selection"], "candidate selection")
        ranks.append(rank)
    winner = max(range(len(ranks)), key=lambda i: ranks[i])
    require(
        ranks[winner][0] and selection["selected"] == selection["candidates"][winner],
        "selected candidate does not win the recomputed gate and ranking",
    )
    for name in (
        "scripts/select_curriculum_checkpoint.py",
        "scripts/declare_workflow_curriculum.py",
        "scripts/build_workflow_skill_curriculum.py",
        "scripts/verify_workflow_skill_data.py",
        "scripts/curriculum_release_checks.py",
        "scripts/build_workspace_calibration.py",
        "scripts/verify_workspace_calibration_data.py",
    ):
        sources[name] = digest(name)
    sources.update(selection_protocol["source_files"])
    require(
        data_protocol["generator_sha256"] == sources["scripts/build_workspace_calibration.py"]
        and data_audit["source_sha256"] == sources["scripts/verify_workspace_calibration_data.py"],
        "calibration construction source changed",
    )
    for path, expected in sources.items():
        require(digest(path) == expected, "source changed: " + path)
    evidence.update(sources)
    return {
        "release_extensions": expected_extension,
        "evidence_files": evidence,
        "source_files": sources,
        "details": {
            "one_training_seed": True,
            "original_noncalibration_splits_preserved": True,
            "development_metrics_and_selection_recomputed": True,
            "skill_queries_do_not_enter_selection": True,
            "calibration_labels_changed": True,
            "same_forward_count_but_not_same_token_budget": True,
        },
    }


def collect_curriculum_evidence(
    checkpoint, selection_path, records, calibration_report, policy_report
):
    result = collect_curriculum_study(selection_path, records)
    evidence, sources = result["evidence_files"], result["source_files"]

    def read(path):
        evidence[str(path)] = digest(path)
        return read_plain(path)

    checkpoint = Path(checkpoint)
    config_path = Path("configs/workflow-curriculum-study-v12.json")
    config = read(config_path)
    design_path = Path(config["fresh_calibration_design"])
    design, origin = read(design_path), read(read_plain(design_path)["study_protocol"])
    selection, calibration, policy = (
        read(selection_path),
        read(calibration_report),
        read(policy_report),
    )
    manifest = read(checkpoint / "manifest.json")
    weights, manifest_hash = (
        digest(checkpoint / "head.safetensors"),
        digest(checkpoint / "manifest.json"),
    )
    require(
        manifest["training"]["release_extensions"] == result["release_extensions"]
        and weights == policy["weights_sha256"] == selection["weights_sha256"]
        and manifest_hash == policy["manifest_sha256"] == calibration["calibrated_manifest_sha256"]
        and policy["selection_sha256"] == digest(selection_path)
        and policy["study_sha256"] == digest(config_path)
        and policy["design_sha256"] == digest(design_path)
        and policy["fresh_calibration_report_sha256"] == digest(calibration_report)
        and policy["policy_refitted"] is False,
        "calibrated checkpoint or policy regression identity changed",
    )
    old_path = Path("runs/workflow-v12-evaluation/calibration/evaluation.json")
    old = read(old_path)
    regression_root = Path(policy_report).parent / "previous-calibration"
    regression = read(regression_root / "evaluation.json")
    prediction_path = regression_root / "predictions.jsonl"
    evidence[str(prediction_path)] = digest(prediction_path)
    require(
        digest(old_path)
        == origin["trigger"]["evaluation_sha256"]
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
        "original calibration regression changed or refitted the policy",
    )
    source_records = {}
    for line in Path(config["records"]).open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] == "calibration":
            record = TrainingRecord.model_validate(raw)
            source_records[record.id] = record
    predictions = [
        json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    require(
        len({r["id"] for r in predictions}) == len(predictions)
        and {r["id"] for r in predictions} == set(source_records),
        "regression population changed",
    )
    for row in predictions:
        record = source_records[row["id"]]
        name = row["question"]
        require(
            row["group"] == record.group_id and row["type"] == record.request.questions[name].type,
            "regression row identity changed",
        )
        gold = next((tag[9:] for tag in record.tags if tag.startswith("gold_key:")), None)
        if "teacher_distribution" in record.tags and gold is not None:
            error = float(row["prediction"] != gold)
        else:
            error = 1 - record.targets[name][row["prediction"]]
        require(
            abs(error - row["expected_error"]) <= 1e-9,
            "regression error disagrees with preserved targets",
        )
    gate, details = design["additional_mandatory_gate"], {}
    for kind in QUESTION_TYPES:
        groups = set(old["fitting"][kind]["policy_group_ids"])
        selected = [r for r in predictions if r["type"] == kind and r["group"] in groups]
        expected_ids = {
            r.id
            for r in source_records.values()
            if r.group_id in groups and next(iter(r.request.questions.values())).type == kind
        }
        require(
            len(selected) == old["fitting"][kind]["questions"]
            and {r["id"] for r in selected} == expected_ids,
            "old policy-fitting population changed",
        )
        accepted = [r for r in selected if not r["abstained"]]
        coverage = len(accepted) / len(selected)
        error = sum(r["expected_error"] for r in accepted) / len(accepted) if accepted else None
        details[kind] = {
            "questions": len(selected),
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
        "scripts/audit_previous_calibration_policy.py",
        "scripts/run_curriculum_release_evaluation.py",
    ):
        sources[filename] = digest(filename)
    require(
        policy["source_sha256"] == sources["scripts/audit_previous_calibration_policy.py"],
        "policy auditor changed",
    )
    evidence.update(sources)
    return {
        "release_extensions": result["release_extensions"],
        "checks": {
            "curriculum_study_provenance": True,
            "previous_calibration_policy_regression": True,
        },
        "details": {
            "curriculum_study_provenance": result["details"],
            "previous_calibration_policy_regression": details,
        },
        "evidence_files": evidence,
        "source_files": sources,
        "previous_policy_report": str(policy_report),
        "previous_policy_report_sha256": digest(policy_report),
    }
