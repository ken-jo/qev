"""Wait for the declared training arms, then advance only through passing release gates."""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    root = Path("runs/workflow-v12-workspace-evaluation")
    checkpoint = Path("checkpoints/veyra-workspace-v12")
    data = Path("data/workflow-v12b/records.jsonl")
    selection_path = Path("runs/workflow-v12-workspace-selection/selection.json")
    root.mkdir(parents=True, exist_ok=False)

    def record(stage, status, **extra):
        result = {
            "stage": stage,
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_allowed": False,
            **extra,
        }
        write(root / "progress.json", result)
        print(json.dumps(result), flush=True)

    def stop(stage, reason):
        record(stage, "gate_failed", reason=reason, final_predictions_started=False)
        raise SystemExit(2)

    def run(stage, *arguments):
        record(stage, "running")
        with (root / (stage + ".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", *map(str, arguments)],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode:
            record(stage, "failed", exit_code=result.returncode, log=str(root / (stage + ".log")))
            raise SystemExit(result.returncode)
        record(stage, "complete")

    record("training", "waiting")
    while True:
        progress = read("runs/workflow-v12-workspace-study/progress.json")
        if progress["status"] == "failed":
            record("training", "failed", training=progress)
            raise SystemExit(1)
        if progress["stage"] == "training" and progress["status"] == "complete":
            break
        time.sleep(10)
    run("selection", "scripts/select_workspace_checkpoint.py")
    selection = read(selection_path)
    if selection["eligible"] is not True:
        stop(
            "selection",
            "No declared candidate passes development and per-type abstention requirements",
        )
    run(
        "calibration",
        "scripts/evaluate_workflow.py",
        "--checkpoint",
        selection["selected"]["checkpoint"],
        "--records",
        data,
        "--split",
        "calibration",
        "--output",
        root / "calibration",
        "--calibrated-output",
        checkpoint,
        "--selection",
        selection_path,
    )
    calibration = read(root / "calibration/evaluation.json")
    if not all(fit["passed"] is True for fit in calibration["fitting"].values()):
        stop("calibration", "Fresh calibration fails the unchanged per-type error/coverage gates")
    run(
        "previous-calibration",
        "scripts/evaluate_workflow.py",
        "--checkpoint",
        checkpoint,
        "--records",
        "data/workflow-v12/records.jsonl",
        "--split",
        "calibration",
        "--output",
        root / "previous-calibration",
        "--selection",
        selection_path,
    )
    run(
        "previous-policy",
        "scripts/audit_previous_calibration_policy.py",
        "--checkpoint",
        checkpoint,
        "--evaluation",
        root / "previous-calibration",
        "--fresh-calibration",
        root / "calibration/evaluation.json",
        "--output",
        root / "previous-calibration-policy.json",
    )
    run(
        "freeze",
        "scripts/freeze_workflow_final.py",
        "--checkpoint",
        checkpoint,
        "--selection",
        selection_path,
        "--records",
        data,
        "--calibration-report",
        root / "calibration/evaluation.json",
        "--previous-policy-report",
        root / "previous-calibration-policy.json",
        "--output",
        root / "final-freeze.json",
    )
    for name, model in (("baseline", "checkpoints/veyra-foundation-v11"), ("final", checkpoint)):
        options = ["--baseline"] if name == "baseline" else ["--selection", selection_path]
        run(
            name,
            "scripts/evaluate_workflow.py",
            "--checkpoint",
            model,
            "--records",
            data,
            "--split",
            "test",
            "--output",
            root / name,
            "--final-freeze",
            root / "final-freeze.json",
            *options,
        )
    run(
        "paired",
        "scripts/compare_workflow_final.py",
        "--before",
        root / "baseline/predictions.jsonl",
        "--after",
        root / "final/predictions.jsonl",
        "--output",
        root / "paired.json",
    )
    for name, records in (
        ("official-regression", "data/foundation-v11-typed-regression/records.jsonl"),
        ("legacy-regression", "data/policy-v8/records.jsonl"),
        ("foundation-regression", "data/foundation-v11/records.jsonl"),
    ):
        run(
            name,
            "scripts/evaluate_foundation.py",
            "--checkpoint",
            checkpoint,
            "--records",
            records,
            "--split",
            "test",
            "--output",
            root / name,
            "--legacy-regression",
        )
    run(
        "runtime",
        "scripts/validate_model.py",
        "--checkpoint",
        checkpoint,
        "--records",
        "data/starter-v2/records.jsonl",
        "--output",
        root / "runtime.json",
    )
    run(
        "photo-http",
        "scripts/benchmark_foundation_http.py",
        "--checkpoint",
        checkpoint,
        "--records",
        "data/foundation-v11/records.jsonl",
        "--output",
        root / "photo-http.json",
    )
    run(
        "network-contract",
        "scripts/verify_workflow_contract.py",
        "--checkpoint",
        checkpoint,
        "--output",
        root / "network-contract.json",
    )
    run(
        "acceptance",
        "scripts/audit_workflow_release.py",
        "--checkpoint",
        checkpoint,
        "--evaluation",
        root,
        "--output",
        root / "release-acceptance.json",
    )
    record(
        "evaluation",
        "complete",
        release_allowed=True,
        published=False,
        remaining="Build the matching runtime wheel, update the HF card and verify the HF package",
    )


if __name__ == "__main__":
    main()
