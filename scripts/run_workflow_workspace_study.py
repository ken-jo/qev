"""Run the two separately declared continuation arms sequentially on the local GPU."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write


def main():
    config_path = Path("configs/workflow-workspace-study-v12.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path("runs/workflow-v12-workspace-study")
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

    for arm in config["execution_order"]:
        record(arm, "running")
        with (root / (arm + ".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "scripts/train_workflow_workspace.py",
                    "--config",
                    str(config_path),
                    "--arm",
                    arm,
                    "--output",
                    str(root / arm),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode:
            record(arm, "failed", exit_code=result.returncode, log=str(root / (arm + ".log")))
            raise SystemExit(result.returncode)
        record(arm, "complete")
    record(
        "training",
        "complete",
        remaining="Merged development, fresh calibration, final evaluation and release checks",
    )


if __name__ == "__main__":
    main()
