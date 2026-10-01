"""Apply the fixed error and coverage requirements to an unchanged frozen policy."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from depth_policy_checks import (
    bootstrap_policy,
    digest,
    policy_population,
    read,
    require,
)
from recovery_policy_checks import DESIGN_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("validation-1", "validation-2", "previous"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "policy audit already exists")
    config, freeze, selection = read(DESIGN_PATH), read(args.policy_freeze), read(args.selection)
    manifest, evaluation = (
        read(args.checkpoint / "manifest.json"),
        read(args.evaluation / "evaluation.json"),
    )
    require(
        freeze["study_sha256"] == digest(DESIGN_PATH)
        and freeze["selection_sha256"] == digest(args.selection)
        and freeze["calibration_report_sha256"] == digest(args.calibration_report)
        and freeze["weights_sha256"] == digest(args.checkpoint / "head.safetensors")
        and freeze["manifest_sha256"]
        == digest(args.checkpoint / "manifest.json")
        == evaluation["protocol"]["manifest_sha256"]
        and datetime.fromisoformat(evaluation["protocol"]["started_at_utc"])
        >= datetime.fromisoformat(freeze["frozen_at_utc"]),
        "evaluation preceded policy freeze or changed the model",
    )
    for path, expected in freeze["source_files"].items():
        require(digest(path) == expected, "policy inference/audit source changed")
    groups = None
    if args.kind != "previous":
        spec = selection["independent_policy_validation_records"][args.kind]
        records = Path(spec["path"])
        require(digest(records) == spec["sha256"], "validation data changed")
        require(
            freeze["validation_records"] == selection["independent_policy_validation_records"],
            "validation binding changed",
        )
    else:
        for role in config["validation_populations"]:
            validation = read(args.policy_freeze.parent / (role + "-policy.json"))
            require(validation["passed"] is True, "an independent validation did not pass")
        old_path = config["previous_group_regression"]["population_report"]
        require(
            digest(old_path) == config["previous_group_regression"]["population_report_sha256"],
            "original group selection changed",
        )
        original = read(old_path)
        groups = {k: set(v["policy_group_ids"]) for k, v in original["fitting"].items()}
        records = Path(config["original_records"])
        require(digest(records) == config["original_records_sha256"], "previous records changed")
    details, rows = policy_population(args.evaluation, records, manifest, args.selection, groups)
    if args.kind == "previous":
        require(
            all(details[k]["questions"] == v["questions"] for k, v in original["fitting"].items()),
            "original policy counts changed",
        )
    result = {
        "passed": all(v["passed"] for v in details.values()),
        "kind": args.kind,
        "by_type": details,
        "bootstrap": {kind: bootstrap_policy(population) for kind, population in rows.items()},
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(__file__),
        "study_sha256": digest(DESIGN_PATH),
        "selection_sha256": digest(args.selection),
        "weights_sha256": freeze["weights_sha256"],
        "manifest_sha256": freeze["manifest_sha256"],
        "policy_freeze_sha256": digest(args.policy_freeze),
        "fresh_calibration_report_sha256": digest(args.calibration_report),
        "evaluation_sha256": digest(args.evaluation / "evaluation.json"),
        "predictions_sha256": digest(args.evaluation / "predictions.jsonl"),
        "policy_refitted": False,
        "population_previously_inspected": args.kind == "previous",
        "release_allowed": False,
        "final_predictions_used": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": result["passed"], "kind": args.kind, "by_type": details}))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
