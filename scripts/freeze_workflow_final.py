"""Freeze an eligible calibrated candidate before observing any final prediction."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write

from veyra.constants import QUESTION_TYPES
from veyra.data import read_records


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/workflow-v12/records.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-policy-report", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("final freeze is immutable")
    config_path = Path("configs/workflow-release-v12.json")
    config, selected, cal = read(config_path), read(args.selection), read(args.calibration_report)
    manifest = read(args.checkpoint / "manifest.json")
    weights = digest(args.checkpoint / "head.safetensors")
    manifest_hash = digest(args.checkpoint / "manifest.json")
    if not (
        selected["eligible"] is True
        and selected["final_or_calibration_used"] is False
        and selected["weights_sha256"] == cal["protocol"]["weights_sha256"] == weights
        and manifest["weights_sha256"] == weights
        and selected["manifest_sha256"] == cal["protocol"]["manifest_sha256"]
        and cal["calibrated_manifest_sha256"] == manifest_hash
        and selected["records_sha256"] == cal["protocol"]["records_sha256"] == digest(args.records)
        and selected["release_protocol_sha256"]
        == cal["protocol"]["release_protocol_sha256"]
        == digest(config_path)
        and cal["protocol"]["arguments"]["split"] == "calibration"
        and cal["protocol"]["selection_sha256"] == digest(args.selection)
    ):
        raise ValueError("selection, calibration, model or dataset provenance mismatch")
    fits = cal["fitting"]
    if set(cal["serialized_policy_check"]) != set(QUESTION_TYPES) or not all(
        item["matches_fitted_policy"] is True for item in cal["serialized_policy_check"].values()
    ):
        raise ValueError("runtime probability serialization did not reproduce the fitted policy")
    gate = config["gates"]["uncertainty"]
    if set(fits) != set(QUESTION_TYPES) or not all(
        fit["passed"] is True
        and fit["always_abstain"] is False
        and fit["coverage"] >= gate["calibration_minimum_coverage"]
        and fit["expected_error"] <= gate["calibration_expected_error_max"]
        for fit in fits.values()
    ):
        raise ValueError("calibration cannot meet the fixed error and coverage requirements")
    temperature_groups = set().union(*(set(x["temperature_group_ids"]) for x in fits.values()))
    policy_groups = set().union(*(set(x["policy_group_ids"]) for x in fits.values()))
    groups, families, counts = defaultdict(set), defaultdict(set), Counter()
    for record in read_records(args.records):
        groups[record.group_id].add(record.split)
        counts[record.split] += 1
        if "domain:workflow_new" in record.tags:
            families[record.split].add(record.family)
    calibration_groups = {group for group, splits in groups.items() if splits == {"calibration"}}
    if (
        temperature_groups & policy_groups
        or temperature_groups | policy_groups != calibration_groups
        or any(len(splits) != 1 for splits in groups.values())
        or families["test"] & (families["train"] | families["dev"] | families["calibration"])
    ):
        raise ValueError("group or final-family separation failed")
    sources = [
        Path("scripts/evaluate_workflow.py"),
        Path("scripts/compare_workflow_final.py"),
        Path("scripts/evaluate_foundation.py"),
        Path("scripts/benchmark_foundation_http.py"),
        Path("scripts/validate_model.py"),
        Path("scripts/verify_workflow_contract.py"),
        Path("scripts/audit_workflow_release.py"),
        Path("reports/workflow-v12/source-snapshots/build_workflow_v12.py.txt"),
        *sorted(Path("src/veyra").glob("*.py")),
    ]
    continuation = None
    if selected.get("release_extensions"):
        if not args.previous_policy_report:
            raise ValueError(
                "continuation requires a passing unchanged-policy regression before final"
            )
        from workflow_extension_checks import collect_release_extension_evidence

        continuation = collect_release_extension_evidence(
            args.checkpoint,
            args.selection,
            args.records,
            args.calibration_report,
            args.previous_policy_report,
        )
    result = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(Path(__file__)),
        "checkpoint": str(args.checkpoint),
        "weights_sha256": weights,
        "manifest_sha256": manifest_hash,
        "records_sha256": digest(args.records),
        "release_protocol_sha256": digest(config_path),
        "selection": str(args.selection),
        "selection_sha256": digest(args.selection),
        "calibration_report": str(args.calibration_report),
        "calibration_report_sha256": digest(args.calibration_report),
        "split_question_counts": dict(counts),
        "new_workflow_families": {key: sorted(value) for key, value in families.items()},
        "calibration_groups_disjoint": True,
        "final_families_disjoint": True,
        "source_files": {str(path): digest(path) for path in sources},
        "regression_datasets": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in {
                "official-regression": Path("data/foundation-v11-typed-regression/records.jsonl"),
                "legacy-regression": Path("data/policy-v8/records.jsonl"),
                "foundation-regression": Path("data/foundation-v11/records.jsonl"),
            }.items()
        },
        "final_predictions_observed": False,
        "required_priorities": [1, 2],
        "continuation": continuation,
    }
    if continuation:
        result["source_files"].update(continuation["source_files"])
    write(args.output, result)
    print(json.dumps({"frozen": str(args.output), "weights_sha256": weights}), flush=True)


if __name__ == "__main__":
    main()
