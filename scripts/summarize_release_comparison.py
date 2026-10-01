"""Audit paired benchmark identities and export aggregate release evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    evaluations, observations, evidence = {}, {}, {}
    for name in ("qev", "laya-base", "laya-specialist"):
        evaluation_path = args.runs / name / "evaluation.json"
        prediction_path = args.runs / name / "predictions.jsonl"
        evaluation = json.loads(evaluation_path.read_text("utf-8"))
        if not evaluation["passed"] or not evaluation["all_requests_evaluated"]:
            raise ValueError("Incomplete benchmark")
        if digest(prediction_path) != evaluation["predictions_sha256"]:
            raise ValueError("Prediction file changed")
        evaluations[name] = evaluation
        observations[name] = {
            row["id"]: row
            for row in map(json.loads, prediction_path.read_text("utf-8").splitlines())
        }
        if len(observations[name]) != 2800:
            raise ValueError("Wrong number of distinct benchmark decisions")
        for suite in evaluation["suites"].values():
            if any(
                suite["truncation_counts"].get(k, 0) for k in ("state", "options", "instructions")
            ):
                raise ValueError("Comparison has truncated evidence")
        evidence[name] = {
            "evaluation_sha256": digest(evaluation_path),
            "predictions_sha256": digest(prediction_path),
        }
        (args.output / (name + ".json")).write_bytes(evaluation_path.read_bytes())
    if len({v["protocol"]["requests_sha256"] for v in evaluations.values()}) != 1:
        raise ValueError("Models saw different raw requests")
    candidate = observations["qev"]
    pairs = {}
    for reference in ("laya-base", "laya-specialist"):
        baseline = observations[reference]
        if set(candidate) != set(baseline):
            raise ValueError("Evaluation identities differ")
        for key, row in candidate.items():
            other = baseline[key]
            if any(
                row[field] != other[field]
                for field in ("gold", "targets", "group", "type", "suite")
            ):
                raise ValueError("Labels or target distributions differ")
        pairs[reference] = {}
        for suite in evaluations["qev"]["suites"]:
            selected = [row for row in candidate.values() if row["suite"] == suite]
            groups = sorted({row["group"] for row in selected})
            totals = np.array(
                [
                    [
                        sum(
                            int(row["correct"]) - int(baseline[row["id"]]["correct"])
                            for row in selected
                            if row["group"] == group
                        ),
                        sum(row["group"] == group for row in selected),
                    ]
                    for group in groups
                ]
            )
            rng = np.random.default_rng(20261001)
            samples = totals[rng.integers(len(groups), size=(10000, len(groups)))].sum(1)
            interval = np.quantile(samples[:, 0] / samples[:, 1], [0.025, 0.975]).tolist()
            pairs[reference][suite] = {
                "qev_minus_reference_accuracy": float(totals[:, 0].sum() / totals[:, 1].sum()),
                "paired_group_bootstrap_95_ci": interval,
                "groups": len(groups),
                "questions": len(selected),
                "bootstrap_replicates": 10000,
                "seed": 20261001,
                "descriptive_not_multiple_comparison_adjusted": True,
                "qev_only_correct": sum(
                    r["correct"] and not baseline[r["id"]]["correct"] for r in selected
                ),
                "reference_only_correct": sum(
                    not r["correct"] and baseline[r["id"]]["correct"] for r in selected
                ),
            }
    inputs = args.runs / "frozen-inputs/protocol.json"
    (args.output / "input-protocol.json").write_bytes(inputs.read_bytes())
    result = {
        "passed": True,
        "models": evidence,
        "paired_accuracy": pairs,
        "requests_sha256": evaluations["qev"]["protocol"]["requests_sha256"],
        "input_protocol_sha256": digest(inputs),
        "aggregator_sha256": digest(Path(__file__)),
        "same_cases_questions_labels_and_metrics": True,
        "all_input_truncation_counts_zero": True,
        "new_training_or_test_temperature_fitting": False,
        "scope": (
            "Paired release regression and two task-held-out English suites, not general parity"
        ),
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(
        json.dumps(
            {
                name: {
                    suite: value["overall"]["accuracy"] for suite, value in report["suites"].items()
                }
                for name, report in evaluations.items()
            },
            indent=2,
        )
    )
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
