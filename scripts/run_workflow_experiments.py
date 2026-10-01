"""Run the declared training stages sequentially on one GPU, preserving logs and failures."""

import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    output = Path("runs/workflow-v12-sequence")
    output.mkdir(parents=True, exist_ok=False)

    def record(stage, status, **extra):
        value = {"stage": stage, "status": status, **extra}
        (output / "progress.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(value), flush=True)

    record("feature_cache", "waiting")
    deadline = time.monotonic() + 1800
    while not Path("data/features-workflow-v12/manifest.json").is_file():
        if time.monotonic() >= deadline:
            record("feature_cache", "failed", reason="Feature cache did not complete in 30 minutes")
            raise SystemExit(1)
        time.sleep(5)
    stages = [
        ("head", ["scripts/train_workflow_head.py"]),
        (
            "backbone-low",
            [
                "scripts/train_workflow_backbone.py",
                "--learning-rate",
                "0.00001",
                "--output",
                "runs/workflow-v12-backbone/low",
            ],
        ),
        (
            "backbone-high",
            [
                "scripts/train_workflow_backbone.py",
                "--learning-rate",
                "0.00003",
                "--output",
                "runs/workflow-v12-backbone/high",
            ],
        ),
    ]
    for name, arguments in stages:
        record(name, "running")
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", *arguments], stdout=log, stderr=subprocess.STDOUT, text=True
            )
        if result.returncode:
            record(name, "failed", exit_code=result.returncode, log=str(output / (name + ".log")))
            raise SystemExit(result.returncode)
        record(name, "complete")
    record(
        "training",
        "complete",
        release_allowed=False,
        remaining="Merged development selection, calibration, independent final and release gates",
    )


if __name__ == "__main__":
    main()
