"""Build prospectively declared independent validation and fresh final populations."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    design_path = Path("configs/workflow-recovery-release-v13.json")
    design = read(design_path)
    root = Path("runs/workflow-recovery-data-v13")
    if root.exists():
        raise FileExistsError("Data preparation outputs are immutable")
    sources = [
        Path(__file__),
        Path("scripts/prepare_followup_calibration_sources.py"),
        Path("scripts/build_followup_calibration.py"),
        Path("scripts/verify_followup_calibration_data.py"),
        Path("scripts/build_recovery_final.py"),
        Path("scripts/verify_recovery_final_data.py"),
        Path("scripts/build_workflow_v12.py"),
        Path("src/veyra/workflow_facts.py"),
    ]
    hashes = {str(p): digest(p) for p in sources}
    write(
        root / "execution-design.json",
        {
            "declared_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_design_sha256": digest(design_path),
            "source_files": hashes,
            "model_predictions_used": False,
            "release_allowed": False,
        },
    )
    env = os.environ.copy()
    runtime = Path(".cache/followup-data-runtime").resolve()
    if not (runtime / "pyarrow").is_dir():
        raise FileNotFoundError("The existing isolated data runtime is required")
    env["PYTHONPATH"] = str(runtime)
    populations = [
        (f"validation-{i + 1}", item) for i, item in enumerate(design["validation_populations"])
    ]
    populations.append(
        (
            "final",
            {
                "design": design["final_data_design"],
                "design_sha256": design["final_data_design_sha256"],
            },
        )
    )
    outputs = {}
    for role, item in populations:
        config_path = Path(item["design"])
        if digest(config_path) != item["design_sha256"]:
            raise ValueError("Population design changed")
        steps = [
            ("sources", "prepare_followup_calibration_sources.py"),
            (
                "build",
                "build_recovery_final.py" if role == "final" else "build_followup_calibration.py",
            ),
            (
                "audit",
                "verify_recovery_final_data.py"
                if role == "final"
                else "verify_followup_calibration_data.py",
            ),
        ]
        for stage, script in steps:
            progress = {"role": role, "stage": stage, "status": "running", "release_allowed": False}
            write(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
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
                    env=env,
                    check=False,
                )
            if result.returncode:
                write(
                    root / "progress.json",
                    {**progress, "status": "failed", "exit_code": result.returncode},
                )
                raise SystemExit(result.returncode)
        folder = Path(read(config_path)["output"])
        audit = read(folder / "independent-target-audit.json")
        if not audit["passed"] or audit["records_sha256"] != digest(folder / "records.jsonl"):
            raise ValueError("Independent data audit failed")
        outputs[role] = {
            str(folder / name): digest(folder / name)
            for name in ("records.jsonl", "protocol.json", "independent-target-audit.json")
        }
    if any(digest(path) != sha for path, sha in hashes.items()):
        raise ValueError("Preparation source changed during execution")
    write(
        root / "complete.json",
        {
            "completed": True,
            "outputs": outputs,
            "release_design_sha256": digest(design_path),
            "source_files": hashes,
            "model_predictions_used": False,
            "release_allowed": False,
        },
    )
    write(root / "progress.json", {"status": "complete", "release_allowed": False})


if __name__ == "__main__":
    main()
