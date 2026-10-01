"""Explain a failed calibration gate from its immutable predictions; change no policy."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from train_foundation_head import write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("calibration diagnosis is immutable")
    evaluation = read(args.calibration / "evaluation.json")
    predictions = args.calibration / "predictions.jsonl"
    if (
        evaluation["protocol"]["arguments"]["split"] != "calibration"
        or digest(predictions) != evaluation["predictions_sha256"]
    ):
        raise ValueError("expected immutable calibration predictions")
    if all(fit["passed"] for fit in evaluation["fitting"].values()):
        raise ValueError("there is no failed calibration requirement to diagnose")
    gate = read("configs/workflow-release-v12.json")["gates"]["uncertainty"]
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    result = {}
    for kind, fit in evaluation["fitting"].items():
        groups = set(fit["policy_group_ids"])
        selected = [row for row in rows if row["type"] == kind and row["group"] in groups]
        if len(selected) != fit["questions"]:
            raise ValueError("policy population does not match the fitted report")
        selected.sort(
            key=lambda row: (-row["confidence"], hashlib.sha256(row["id"].encode()).hexdigest())
        )
        total, maximum, risk_at_maximum = 0.0, 0, None
        for count, row in enumerate(selected, 1):
            total += row["expected_error"]
            boundary = count == len(selected) or row["confidence"] != selected[count]["confidence"]
            if boundary and count >= 20 and total / count <= gate["calibration_expected_error_max"]:
                maximum, risk_at_maximum = count, total / count
        minimum = max(20, math.ceil(len(selected) * gate["calibration_minimum_coverage"]))
        prefix = selected[:minimum]
        result[kind] = {
            "passed": fit["passed"],
            "policy_questions": len(selected),
            "minimum_required_answers": minimum,
            "fixed_count_error_at_minimum_coverage": sum(row["expected_error"] for row in prefix)
            / minimum,
            "maximum_threshold_coverage_at_error_limit": maximum / len(selected),
            "error_at_maximum_coverage": risk_at_maximum,
            "by_domain_in_fixed_count_prefix": {
                domain: {
                    "questions": len(subset := [r for r in prefix if r["domain"] == domain]),
                    "expected_error": sum(r["expected_error"] for r in subset) / len(subset),
                }
                for domain in sorted({r["domain"] for r in prefix})
            },
        }
    report = {
        "scope": "Post-hoc explanation of calibration failure; not a new fit or performance claim",
        "source_sha256": digest(Path(__file__)),
        "evaluation_sha256": digest(args.calibration / "evaluation.json"),
        "predictions_sha256": digest(predictions),
        "weights_sha256": evaluation["protocol"]["weights_sha256"],
        "by_type": result,
        "policy_or_weights_changed": False,
        "final_predictions_used": False,
        "limitations": (
            "The calibration set is now inspected. Fixed-count prefixes can split confidence "
            "ties; maximum threshold coverage respects ties. Future training/selection must "
            "continue to exclude these groups."
        ),
    }
    write(args.output, report)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
