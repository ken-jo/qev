"""Require the frozen fresh policy to pass the original inspected policy groups too."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write

from veyra.constants import QUESTION_TYPES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--fresh-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-config", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("previous calibration policy audit is immutable")
    design_path = Path("configs/workflow-workspace-calibration-v12.json")
    design = read(design_path)
    origin_study = read(design["study_protocol"])
    study_path = args.study_config or Path(design["study_protocol"])
    study = read(study_path)
    if args.study_config and not (
        Path(args.study_config)
        in {
            Path("configs/workflow-curriculum-study-v12.json"),
            Path("configs/workflow-depth-study-v12.json"),
        }
        and study["fresh_calibration_design_sha256"] == digest(design_path)
        and study["release_protocol_sha256"] == origin_study["release_protocol_sha256"]
        and study["records_sha256"] == design["old_records_sha256"]
    ):
        raise ValueError("undeclared follow-up use of the original calibration regression")
    old_path = Path("runs/workflow-v12-evaluation/calibration/evaluation.json")
    old = read(old_path)
    new_path = args.evaluation / "evaluation.json"
    new, fresh = read(new_path), read(args.fresh_calibration)
    selection_path = Path(fresh["protocol"]["arguments"]["selection"])
    selection = read(selection_path)
    manifest = read(args.checkpoint / "manifest.json")
    weights, manifest_hash = (
        digest(args.checkpoint / "head.safetensors"),
        digest(args.checkpoint / "manifest.json"),
    )
    if not (
        digest(old_path) == origin_study["trigger"]["evaluation_sha256"]
        and selection["eligible"] is True
        and new["protocol"]["selection_sha256"]
        == fresh["protocol"]["selection_sha256"]
        == digest(selection_path)
        and selection["study_protocol_sha256"] == digest(study_path)
        and selection["calibration_design_sha256"] == digest(design_path)
        and fresh["protocol"]["records_sha256"] == digest(Path(design["output"]) / "records.jsonl")
        and new["protocol"]["weights_sha256"] == fresh["protocol"]["weights_sha256"] == weights
        and new["protocol"]["manifest_sha256"]
        == fresh["calibrated_manifest_sha256"]
        == manifest_hash
        and new["protocol"]["records_sha256"]
        == old["protocol"]["records_sha256"]
        == design["old_records_sha256"]
        and new["protocol"]["arguments"]["split"] == "calibration"
        and new["fitting"] is None
        and new["calibrated_manifest_sha256"] is None
        and new["calibration"] == fresh["calibration"] == manifest["calibration"]
        and set(fresh["fitting"]) == set(QUESTION_TYPES)
        and all(fit["passed"] is True for fit in fresh["fitting"].values())
        and set(fresh["serialized_policy_check"]) == set(QUESTION_TYPES)
        and all(
            item["matches_fitted_policy"] is True
            for item in fresh["serialized_policy_check"].values()
        )
        and datetime.fromisoformat(new["protocol"]["started_at_utc"])
        >= datetime.fromisoformat(fresh["protocol"]["started_at_utc"])
    ):
        raise ValueError("policy regression does not bind to the unchanged fresh calibration")
    path = args.evaluation / "predictions.jsonl"
    if digest(path) != new["predictions_sha256"]:
        raise ValueError("policy regression predictions changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("policy regression contains duplicate questions")
    gate = design["additional_mandatory_gate"]
    details = {}
    for kind in QUESTION_TYPES:
        groups = set(old["fitting"][kind]["policy_group_ids"])
        population = [r for r in rows if r["type"] == kind and r["group"] in groups]
        if (
            len(population) != old["fitting"][kind]["questions"]
            or {r["group"] for r in population} != groups
        ):
            raise ValueError("previous policy group population changed")
        accepted = [row for row in population if not row["abstained"]]
        coverage = len(accepted) / len(population)
        error = sum(row["expected_error"] for row in accepted) / len(accepted) if accepted else None
        details[kind] = {
            "questions": len(population),
            "accepted": len(accepted),
            "coverage": coverage,
            "expected_error": error,
            "passed": coverage >= gate["per_type_minimum_coverage"]
            and error is not None
            and error <= gate["per_type_expected_error_max"],
        }
    result = {
        "passed": all(row["passed"] for row in details.values()),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "by_type": details,
        "weights_sha256": weights,
        "manifest_sha256": manifest_hash,
        "policy_refitted": False,
        "population_previously_inspected": True,
        "source_sha256": digest(Path(__file__)),
        "design_sha256": digest(design_path),
        "study_sha256": digest(study_path),
        "selection_sha256": digest(selection_path),
        "original_calibration_report_sha256": digest(old_path),
        "fresh_calibration_report_sha256": digest(args.fresh_calibration),
        "evaluation_sha256": digest(new_path),
        "predictions_sha256": digest(path),
    }
    write(args.output, result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
