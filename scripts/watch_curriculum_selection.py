"""Advance from completed curriculum training to its declared merged development screen."""

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
    output = Path("runs/workflow-v12-curriculum-evaluation")
    output.mkdir(parents=True, exist_ok=False)
    training = Path("runs/workflow-v12-curriculum-study")

    def record(stage, status, **extra):
        progress = {
            "stage": stage,
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_allowed": False,
            "final_predictions_started": False,
            **extra,
        }
        write(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)

    record("training", "waiting")
    while not (training / "complete.json").exists():
        if read(training / "progress.json")["status"] == "failed":
            record("training", "failed")
            raise SystemExit(1)
        time.sleep(15)
    record("selection", "running")
    with (output / "selection.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-u", "scripts/select_curriculum_checkpoint.py"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        record("selection", "failed", exit_code=result.returncode)
        raise SystemExit(result.returncode)
    selection = read("runs/workflow-v12-curriculum-selection/selection.json")
    if not selection["eligible"]:
        record("selection", "gate_failed", reason="No declared candidate meets development gates")
        raise SystemExit(2)
    record(
        "selection",
        "complete",
        next_stage="Fresh calibration, unchanged original-policy regression and full release audit",
    )


if __name__ == "__main__":
    main()
