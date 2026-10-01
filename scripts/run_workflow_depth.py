"""Run the two declared depth arms sequentially on the single GPU."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write


def main():
    config_path = Path("configs/workflow-depth-study-v12.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    output = Path("runs/workflow-v12-depth-study")
    if (output / "complete.json").exists():
        raise FileExistsError("depth training already finished")
    for arm in config["arms"]:
        target = output / arm
        if (target / "complete.json").exists():
            protocol = json.loads((target / "protocol.json").read_text(encoding="utf-8"))
            if protocol["configuration_sha256"] != config_hash:
                raise ValueError("completed arm belongs to a different declaration")
            continue
        if target.exists():
            raise FileExistsError("incomplete arm requires an explicit recovery: " + str(target))
        progress = {
            "arm": arm,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": config_hash,
            "release_allowed": False,
        }
        write(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        with (output / (arm + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", "scripts/train_workflow_depth.py", "--arm", arm],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            progress.update(status="failed", returncode=result.returncode)
            write(output / "progress.json", progress)
            raise RuntimeError("depth arm failed; inspect " + arm + ".log")
    write(
        output / "complete.json",
        {
            "completed": True,
            "configuration_sha256": config_hash,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "arms": config["arms"],
            "release_allowed": False,
        },
    )
    print(json.dumps({"completed": True, "release_allowed": False}), flush=True)


if __name__ == "__main__":
    main()
