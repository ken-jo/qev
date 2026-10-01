"""Compute mandatory release decisions from exact-model evidence, never caller assertions."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write

from veyra.constants import QUESTION_TYPES
from veyra.workflow_release_gate import REQUIRED_CHECKS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=Path("runs/workflow-v12-evaluation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("acceptance evidence is immutable")
    evidence = {}

    def read(path):
        path = Path(path)
        evidence[str(path)] = digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def require(condition, message):
        if not condition:
            raise ValueError("Evidence rejected: " + message)

    root = args.evaluation
    protocol_path = Path("configs/workflow-release-v12.json")
    config = read(protocol_path)
    gates = config["gates"]
    freeze = read(root / "final-freeze.json")
    selection = read(freeze["selection"])
    baseline_dev = read(selection["baseline_merged_dev"])
    cal = read(freeze["calibration_report"])
    baseline = read(root / "baseline/evaluation.json")
    final = read(root / "final/evaluation.json")
    paired = read(root / "paired.json")
    official = read(root / "official-regression/evaluation.json")
    legacy = read(root / "legacy-regression/evaluation.json")
    foundation = read(root / "foundation-regression/evaluation.json")
    runtime = read(root / "runtime.json")
    photo = read(root / "photo-http.json")
    contract = read(root / "network-contract.json")
    manifest = read(args.checkpoint / "manifest.json")
    weights = digest(args.checkpoint / "head.safetensors")
    manifest_hash = digest(args.checkpoint / "manifest.json")
    require(
        freeze["weights_sha256"] == selection["weights_sha256"] == weights
        and freeze["manifest_sha256"] == manifest_hash
        and freeze["release_protocol_sha256"] == digest(protocol_path)
        and freeze["selection_sha256"] == digest(freeze["selection"])
        and freeze["calibration_report_sha256"] == digest(freeze["calibration_report"])
        and selection["baseline_merged_dev_sha256"] == digest(selection["baseline_merged_dev"]),
        "the frozen model, selection or calibration changed",
    )
    require(
        manifest["weights_sha256"] == weights
        and manifest["training"]["intermediate"] is False
        and selection["eligible"] is True
        and selection["final_or_calibration_used"] is False
        and selection["release_protocol_sha256"] == digest(protocol_path),
        "development selection or calibrated checkpoint is incomplete",
    )
    for path, expected in freeze["source_files"].items():
        require(digest(path) == expected, "source changed since final freeze: " + path)
        evidence[path] = expected
    for result in (final, official, legacy, foundation):
        require(
            result["protocol"]["weights_sha256"] == weights
            and result["protocol"]["manifest_sha256"] == manifest_hash
            and result["protocol"]["arguments"]["split"] == "test"
            and result["protocol"]["merged_bf16_deployment"] is True,
            "accuracy evidence describes another checkpoint or split",
        )
    for name, result in (
        ("baseline", baseline),
        ("final", final),
        ("official-regression", official),
        ("legacy-regression", legacy),
        ("foundation-regression", foundation),
    ):
        predictions = root / name / "predictions.jsonl"
        require(digest(predictions) == result["predictions_sha256"], "predictions changed: " + name)
        evidence[str(predictions)] = digest(predictions)
        dataset = Path(result["protocol"]["arguments"]["records"])
        require(digest(dataset) == result["protocol"]["records_sha256"], "dataset changed")
        expected_script = (
            "scripts/evaluate_recovery_final.py"
            if name in {"baseline", "final"}
            else "scripts/evaluate_foundation.py"
        )
        require(result["protocol"]["source_sha256"] == digest(expected_script), "evaluator changed")
        if name in freeze["regression_datasets"]:
            require(
                result["protocol"]["records_sha256"]
                == freeze["regression_datasets"][name]["sha256"],
                "wrong regression dataset",
            )
    baseline_records = Path("data/workflow-v12b/records.jsonl")
    current_records = Path(final["protocol"]["arguments"]["records"])

    def preserved_splits(path):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return {split: [row for row in rows if row["split"] == split] for split in ("train", "dev")}

    preserved = preserved_splits(baseline_records)
    require(
        preserved == preserved_splits(current_records),
        "recovery training/development records differ from the baseline population",
    )
    require(
        current_records == Path(selection["fresh_final_records"]),
        "the independent final population was replaced",
    )
    evidence[str(baseline_records)] = digest(baseline_records)
    evidence[str(current_records)] = digest(current_records)
    require(
        baseline["protocol"]["weights_sha256"] == config["baseline_weights_sha256"]
        and baseline["protocol"]["manifest_sha256"] == config["baseline_manifest_sha256"]
        and baseline_dev["weights_sha256"] == config["baseline_weights_sha256"]
        and baseline_dev["manifest_sha256"] == config["baseline_manifest_sha256"]
        and baseline_dev["records_sha256"] == digest(baseline_records)
        and selection["records_sha256"] == freeze["records_sha256"]
        and baseline["protocol"]["records_sha256"]
        == final["protocol"]["records_sha256"]
        == freeze["records_sha256"]
        and baseline["protocol"]["subset_sha256"] == final["protocol"]["subset_sha256"]
        and baseline["protocol"]["arguments"]["split"] == "test"
        and paired["before_sha256"] == baseline["predictions_sha256"]
        and paired["after_sha256"] == final["predictions_sha256"],
        "paired baseline/final inputs or predictions differ",
    )
    for result in (baseline, final):
        require(
            result["protocol"]["final_freeze_sha256"] == digest(root / "final-freeze.json")
            and result["protocol"]["release_protocol_sha256"] == digest(protocol_path)
            and datetime.fromisoformat(result["protocol"]["started_at_utc"])
            >= datetime.fromisoformat(freeze["frozen_at_utc"]),
            "final evaluation did not follow the recorded freeze",
        )
    require(
        runtime["checkpoint_weights_sha256"]
        == photo["protocol"]["weights_sha256"]
        == contract["weights_sha256"]
        == weights
        and runtime["checkpoint_manifest_sha256"]
        == photo["protocol"]["manifest_sha256"]
        == contract["manifest_sha256"]
        == manifest_hash
        and cal["calibrated_manifest_sha256"] == manifest_hash
        and cal["protocol"]["weights_sha256"] == weights,
        "runtime or calibration evidence describes another model",
    )
    require(
        paired["source_sha256"] == digest("scripts/compare_workflow_final.py")
        and contract["source_sha256"] == digest("scripts/verify_workflow_contract.py")
        and photo["protocol"]["script_sha256"] == digest("scripts/benchmark_foundation_http.py")
        and contract["records_sha256"]
        == photo["protocol"]["dataset_sha256"]
        == freeze["regression_datasets"]["foundation-regression"]["sha256"],
        "comparison, contract or photograph benchmark provenance changed",
    )

    checks, details = {}, {}
    workflow = paired["new_workflow"]
    accuracy = workflow["accuracy"]
    final_families = set(freeze["new_workflow_families"]["test"])
    checks["workflow_generalization"] = bool(
        accuracy["groups"] >= gates["new_workflow"]["minimum_final_groups"]
        and len(workflow["families"]) >= gates["new_workflow"]["minimum_final_families"]
        and set(workflow["families"]) == final_families
        and final["detailed_by_token_length_bin"]
        and any(key != "unannotated" for key in final["detailed_by_boolean_operator_depth"])
        and accuracy["delta_95_ci"][0] > gates["new_workflow"]["paired_accuracy_delta_95_lower_gt"]
    )
    details["workflow_generalization"] = workflow
    uncertainty = paired["uncertainty"]
    ugate = gates["uncertainty"]
    cost = uncertainty["matched_coverage_cost"]
    conditions = config["data"]["uncertainty_conditions"]
    checks["uncertainty_probability"] = bool(
        uncertainty["nll"]["delta"] < ugate["nll_delta_lt"]
        and uncertainty["brier"]["delta"] < ugate["brier_delta_lt"]
        and cost["delta"] < ugate["matched_coverage_expected_cost_delta_lt"]
        and uncertainty["nll"]["delta_95_ci"][1] < ugate["paired_nll_delta_95_upper_lt"]
        and cost["coverage"] == ugate["matched_answer_coverage"]
        and all(
            uncertainty["condition_groups"].get(condition, 0)
            >= ugate["minimum_final_groups_per_condition"]
            for condition in conditions
        )
        and set(uncertainty["conditions"]) == set(conditions)
    )
    details["uncertainty_probability"] = uncertainty
    fits = cal["fitting"]
    temperature_groups = set().union(*(set(x["temperature_group_ids"]) for x in fits.values()))
    policy_groups = set().union(*(set(x["policy_group_ids"]) for x in fits.values()))
    policy = final["risk"]["actual_policy"]
    uncertainty_policy = final["detailed_by_domain"]["uncertainty"]["actual_policy"]
    checks["uncertainty_abstention"] = bool(
        set(fits) == set(QUESTION_TYPES)
        and set(cal["serialized_policy_check"]) == set(QUESTION_TYPES)
        and all(
            item["matches_fitted_policy"] is True
            for item in cal["serialized_policy_check"].values()
        )
        and not temperature_groups & policy_groups
        and all(
            f["passed"] is True
            and f["always_abstain"] is False
            and finite(f["expected_error"])
            and f["expected_error"] <= ugate["calibration_expected_error_max"]
            and f["coverage"] >= ugate["calibration_minimum_coverage"]
            for f in fits.values()
        )
        and policy["coverage"] >= ugate["final_overall_minimum_coverage"]
        and uncertainty_policy["abstention_rate"]
        >= ugate["final_uncertainty_minimum_abstention_rate"]
    )
    details["uncertainty_abstention"] = {
        "calibration": {
            kind: {k: v for k, v in f.items() if not k.endswith("group_ids")}
            for kind, f in fits.items()
        },
        "overall_final": policy,
        "uncertainty_final": uncertainty_policy,
        "calibration_groups_disjoint": not bool(temperature_groups & policy_groups),
        "limitation": "Calibration expected error is empirical; it is not a bound on final risk.",
    }
    retention = gates["retention"]
    changes = {}
    for domain in retention["domains"]:
        before = baseline_dev["metrics"]["by_domain"][domain]["accuracy"]
        after = selection["selected"]["metrics"]["by_domain"][domain]["accuracy"]
        changes[domain] = {"before": before, "after": after, "delta": after - before}
    checks["retention_development"] = all(
        value["delta"] >= -retention["development_max_accuracy_drop"] - 1e-9
        for value in changes.values()
    ) and all(
        domain in paired["retention"] and len(paired["retention"][domain]["delta_95_ci"]) == 2
        for domain in retention["domains"]
    )
    details["retention_development"] = {"development": changes, "final": paired["retention"]}
    for modality in ("text", "image"):
        value = legacy["by_modality"][modality]["accuracy"]
        checks["legacy_" + modality] = (
            finite(value) and value >= retention["legacy_" + modality + "_accuracy_min"]
        )
        details["legacy_" + modality] = legacy["by_modality"][modality]
    value = official["overall"]["accuracy"]
    checks["official_workflow"] = finite(value) and value >= gates["official_workflow_accuracy_min"]
    details["official_workflow"] = official["overall"]
    latency, pp = gates["latency"], photo["protocol"]
    checks["runtime"] = bool(
        photo["requests"] >= latency["photos"]
        and len(photo["latency_ms"]["samples"]) == photo["requests"]
        and finite(photo["latency_ms"]["p95"])
        and photo["latency_ms"]["p95"] <= latency["http_p95_ms_max"]
        and "4060 Ti" in photo["gpu"]
        and 7500 <= photo["gpu_memory_mib"] <= 8500
        and pp["warmups"] >= latency["warmups"]
        and pp["photos_per_request"] == 1
        and pp["questions_per_request"] == latency["questions_per_request"]
        and pp["candidates_per_question"] == latency["candidates_per_question"]
        and pp["resident"] is True
        and pp["feature_cache"] is False
        and pp["serial"] is True
        and set(runtime["protocol"])
        == {
            "choice_id_order_max_probability_difference",
            "choice_id_order_pass",
            "zero_generated_tokens",
            "http_path_escape_rejected",
            "http_token_budget_rejected",
        }
        and all(
            runtime["protocol"][name] is True
            for name in (
                "choice_id_order_pass",
                "zero_generated_tokens",
                "http_path_escape_rejected",
                "http_token_budget_rejected",
            )
        )
        and finite(runtime["protocol"]["choice_id_order_max_probability_difference"])
        and 0 <= runtime["protocol"]["choice_id_order_max_probability_difference"] < 1e-5
    )
    details["runtime"] = {
        "gpu": photo["gpu"],
        "p95_ms": photo["latency_ms"]["p95"],
        "photos": photo["requests"],
        "feature_cache": pp["feature_cache"],
    }
    checks["single_network"] = bool(
        contract["passed"] is True
        and set(contract["results"]) == {"text", "photograph_three_questions"}
        and all(
            r["backbone_forwards"] == 1
            and r["generated_answer_tokens"] == 0
            and r["valid_probabilities"] is True
            for r in contract["results"].values()
        )
    )
    details["single_network"] = contract["results"]
    checks["provenance"] = True  # Every mandatory binding above must succeed before this point.
    require(set(checks) == set(REQUIRED_CHECKS), "incomplete acceptance implementation")
    extensions = manifest["training"].get("release_extensions", {})
    require(set(extensions) == {"workflow_recovery_v13"}, "recovery extension missing")
    if extensions:
        from recovery_policy_checks import collect_recovery_evidence

        continuation = freeze.get("continuation")
        require(continuation is not None, "continuation was not checked before final inference")
        checked = collect_recovery_evidence(
            args.checkpoint,
            freeze["selection"],
            final["protocol"]["arguments"]["records"],
            freeze["calibration_report"],
            continuation["previous_policy_report"],
        )
        require(checked == continuation, "continuation evidence changed after final freeze")
        checks.update(checked["checks"])
        details.update(checked["details"])
        evidence.update(checked["evidence_files"])
    result = {
        "release_allowed": all(checks.values()),
        "recovery_population_binding": {
            "preserved_split_questions": {k: len(v) for k, v in preserved.items()},
            "fresh_final_records": str(current_records),
            "previous_final_result_preserved": True,
            "original_numerical_thresholds_unchanged": True,
        },
        "required_priorities": config["required_priorities"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(Path(__file__)),
        "release_protocol_sha256": digest(protocol_path),
        "weights_sha256": weights,
        "manifest_sha256": manifest_hash,
        "release_extensions": extensions,
        "checks": checks,
        "details": details,
        "evidence_files": evidence,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "foundation_regression": foundation["by_domain"],
        "scope": (
            "Controlled procedural transfer plus public-data regressions; "
            "not unrestricted operational reliability."
        ),
        "published": False,
    }
    write(args.output, result)
    print(json.dumps({"release_allowed": result["release_allowed"], "checks": checks}), flush=True)
    if not result["release_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
