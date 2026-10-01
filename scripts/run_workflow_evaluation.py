"""Continue the frozen experiment through selection, calibration and all release evidence."""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/workflow-v12-evaluation"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/veyra-workflow-v12"))
    args = parser.parse_args()
    root, checkpoint = args.output, args.checkpoint
    root.mkdir(parents=True, exist_ok=False)

    def record(stage, status, **extra):
        result = {
            "stage": stage,
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        write(root / "progress.json", result)
        print(json.dumps(result), flush=True)

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
            record(
                stage,
                "blocked",
                exit_code=result.returncode,
                release_allowed=False,
                log=str(root / (stage + ".log")),
            )
            raise SystemExit(result.returncode)
        record(stage, "complete")

    record("training", "waiting")
    while True:
        progress = read("runs/workflow-v12-sequence/progress.json")
        if progress["status"] == "failed":
            record("training", "blocked", training=progress, release_allowed=False)
            raise SystemExit(1)
        if progress["stage"] == "training" and progress["status"] == "complete":
            break
        time.sleep(10)
    selection_path = Path("runs/workflow-v12-selection/selection.json")
    run("selection", "scripts/select_workflow_checkpoint.py")
    selection = read(selection_path)
    if selection["eligible"] is not True:
        record(
            "selection",
            "blocked",
            reason="No candidate meets fixed development requirements",
            release_allowed=False,
        )
        raise SystemExit(2)
    run(
        "calibration",
        "scripts/evaluate_workflow.py",
        "--checkpoint",
        selection["selected"]["checkpoint"],
        "--split",
        "calibration",
        "--output",
        root / "calibration",
        "--calibrated-output",
        checkpoint,
        "--selection",
        selection_path,
    )
    run(
        "freeze",
        "scripts/freeze_workflow_final.py",
        "--checkpoint",
        checkpoint,
        "--selection",
        selection_path,
        "--calibration-report",
        root / "calibration/evaluation.json",
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
        remaining="Update model documentation and build/verify the HF preparation package",
    )


if __name__ == "__main__":
    main()
