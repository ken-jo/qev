"""Build and audit the prospectively fixed policy-fitting and validation populations."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/workflow-v12-depth-policy-data"))
    args = parser.parse_args()
    study_path = Path("configs/workflow-depth-policy-v12.json")
    study = read(study_path)
    for path, expected in study["source_files"].items():
        if digest(path) != expected:
            raise ValueError("declared preparation source changed: " + path)
    for path, expected in study["frozen_generator_ancestors"].items():
        if digest(path) != expected:
            raise ValueError("historical generator changed: " + path)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    write(
        root / "protocol.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "study_sha256": digest(study_path),
            "source_sha256": digest(__file__),
            "model_predictions_used": False,
        },
    )
    outputs = {}
    for role, key in (("fit", "fitting_data_design"), ("validation", "validation_data_design")):
        config_path = Path(study[key])
        if digest(config_path) != study[key + "_sha256"]:
            raise ValueError("data design changed")
        for stage, script in (
            ("sources", "prepare_followup_calibration_sources.py"),
            ("build", "build_followup_calibration.py"),
            ("audit", "verify_followup_calibration_data.py"),
        ):
            progress = {"role": role, "stage": stage, "status": "running", "release_allowed": False}
            write(root / "progress.json", progress)
            with (root / f"{role}-{stage}.log").open("x", encoding="utf-8") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "utf8",
                        str(Path("scripts") / script),
                        "--config",
                        str(config_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                write(
                    root / "progress.json",
                    {**progress, "status": "failed", "exit_code": result.returncode},
                )
                raise SystemExit(result.returncode)
        output = Path(read(config_path)["output"])
        audit = read(output / "independent-target-audit.json")
        if not audit["passed"] or audit["records_sha256"] != digest(output / "records.jsonl"):
            raise ValueError("independent data audit mismatch")
        outputs[role] = {
            str(output / filename): digest(output / filename)
            for filename in ("records.jsonl", "protocol.json", "independent-target-audit.json")
        }
    result = {
        "completed": True,
        "study_sha256": digest(study_path),
        "outputs": outputs,
        "model_predictions_used": False,
        "release_allowed": False,
    }
    write(root / "complete.json", result)
    write(root / "progress.json", {"status": "complete", "release_allowed": False})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
