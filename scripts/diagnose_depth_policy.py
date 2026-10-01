"""Explain the failed depth policy using saved calibration outputs; change no model or policy."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows, threshold):
    accepted = [r for r in rows if r["confidence"] >= threshold]
    return {
        "questions": len(rows),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(rows),
        "expected_error": (
            sum(r["expected_error"] for r in accepted) / len(accepted) if accepted else None
        ),
        "threshold": threshold,
    }


def threshold_fit(rows, maximum_error, minimum_coverage=0.6):
    minimum = max(20, math.ceil(len(rows) * minimum_coverage))
    for threshold in sorted({r["confidence"] for r in rows}):
        value = summarize(rows, threshold)
        if value["accepted"] >= minimum and value["expected_error"] <= maximum_error:
            return value
    return None


def population_diagnosis(rows, threshold):
    ordered = sorted(rows, key=lambda r: (-r["confidence"], r["id"]))
    at_coverages = {}
    for coverage in (0.6, 0.7, 0.8, 0.9, 1.0):
        index = math.ceil(len(rows) * coverage) - 1
        at_coverages[str(coverage)] = summarize(rows, ordered[index]["confidence"])
    groups = Counter(r["group"] for r in rows)
    return {
        "questions": len(rows),
        "groups": len(groups),
        "questions_per_group": dict(sorted(Counter(groups.values()).items())),
        "current_policy": summarize(rows, threshold),
        "confidence_prefixes_including_boundary_ties": at_coverages,
        "maximum_coverage_under_15pct_error": threshold_fit(rows, 0.15, 0.0),
        "current_policy_by_domain": {
            domain: summarize([r for r in rows if r["domain"] == domain], threshold)
            for domain in sorted({r["domain"] for r in rows})
        },
        "current_policy_by_condition": {
            condition: summarize([r for r in rows if r["condition"] == condition], threshold)
            for condition in sorted({r["condition"] for r in rows})
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("runs/workflow-v12-depth-policy-diagnosis/report.json")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnosis is immutable")
    root = Path("runs/workflow-v12-depth-release")
    fresh_path = root / "calibration/evaluation.json"
    old_path = root / "previous-calibration/evaluation.json"
    original_path = Path("runs/workflow-v12-evaluation/calibration/evaluation.json")
    regression_path = root / "previous-calibration-policy.json"
    fresh, old, original, regression = map(
        read, (fresh_path, old_path, original_path, regression_path)
    )
    if not (
        regression["passed"] is False
        and regression["policy_refitted"] is False
        and regression["fresh_calibration_report_sha256"] == digest(fresh_path)
        and regression["evaluation_sha256"] == digest(old_path)
        and regression["original_calibration_report_sha256"] == digest(original_path)
        and fresh["protocol"]["weights_sha256"] == old["protocol"]["weights_sha256"]
        and fresh["calibrated_manifest_sha256"] == old["protocol"]["manifest_sha256"]
        and fresh["calibration"] == old["calibration"]
        and old["fitting"] is None
    ):
        raise ValueError("calibration lineage changed")
    files = {str(p): digest(p) for p in (fresh_path, old_path, original_path, regression_path)}
    populations = {}
    for name, folder, evaluation, fits in (
        ("fresh", "calibration", fresh, fresh["fitting"]),
        ("previous", "previous-calibration", old, original["fitting"]),
    ):
        path = root / folder / "predictions.jsonl"
        if digest(path) != evaluation["predictions_sha256"]:
            raise ValueError("saved predictions changed")
        files[str(path)] = digest(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate prediction IDs")
        populations[name] = {}
        for kind, fit in fits.items():
            groups = set(fit["policy_group_ids"])
            subset = [r for r in rows if r["type"] == kind and r["group"] in groups]
            if len(subset) != fit["questions"] or {r["group"] for r in subset} != groups:
                raise ValueError("policy-group population changed")
            if any(not 0 <= r["expected_error"] <= 1 for r in subset):
                raise ValueError("expected error is outside its declared bounded range")
            populations[name][kind] = subset
    results = {}
    for kind, fit in fresh["fitting"].items():
        current = fit["threshold"]
        groups = {
            name: population_diagnosis(population[kind], current)
            for name, population in populations.items()
        }
        actual = groups["previous"]["current_policy"]
        for key in ("questions", "accepted", "coverage", "expected_error"):
            if abs(actual[key] - regression["by_type"][kind][key]) > 1e-12:
                raise ValueError("previous-policy metric reconstruction disagrees")
        sensitivity = {}
        for budget in (0.10, 0.12, 0.15):
            candidate = threshold_fit(populations["fresh"][kind], budget)
            sensitivity[str(budget)] = {
                "fresh_fit": candidate,
                "previous_transfer": (
                    summarize(populations["previous"][kind], candidate["threshold"])
                    if candidate is not None
                    else None
                ),
            }
        results[kind] = {"populations": groups, "empirical_budget_sensitivity": sensitivity}
    result = {
        "scope": "Post-hoc calibration diagnosis and sensitivity; not a new fitted release policy",
        "source_sha256": digest(__file__),
        "inputs": files,
        "weights_sha256": regression["weights_sha256"],
        "calibrated_manifest_sha256": regression["manifest_sha256"],
        "by_type": results,
        "policy_or_weights_changed": False,
        "final_predictions_used": False,
        "release_allowed": False,
        "limitations": (
            "Both calibration populations were previously inspected. Sensitivity thresholds are "
            "diagnostic only, with no statistical risk guarantee or candidate promotion. Prefixes "
            "include all confidence ties. A subsequent policy method requires a prospective "
            "declaration and separate calibration validation; numerical release gates stay fixed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps({kind: value["empirical_budget_sensitivity"] for kind, value in results.items()})
    )


if __name__ == "__main__":
    main()
