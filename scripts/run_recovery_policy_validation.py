"""Fit one recovery policy and validate it before any new final prediction."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from recovery_policy_checks import collect_admission, collect_recovery_evidence
from train_foundation_head import write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(root):
    checkpoint = Path("checkpoints/veyra-backbone-recovery-v13")
    data = Path("data/workflow-v12-cohort/records.jsonl")
    selection_path = Path("runs/backbone-recovery-policy-selection-v13/selection.json")

    final_started = False

    def record(stage, status, **extra):
        result = {
            "stage": stage,
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_allowed": False,
            "final_predictions_started": final_started,
            **extra,
        }
        write(root / "progress.json", result)
        print(json.dumps(result), flush=True)

    def stop(stage, reason):
        record(stage, "gate_failed", reason=reason)
        raise SystemExit(2)

    def run(stage, *arguments, python=True):
        record(stage, "running")
        with (root / (stage + ".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", *map(str, arguments)]
                if python
                else list(map(str, arguments)),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode:
            record(stage, "failed", exit_code=result.returncode, log=str(root / (stage + ".log")))
            raise SystemExit(result.returncode)
        record(stage, "complete")

    selection = read(selection_path)
    collect_admission(selection_path)
    if checkpoint.exists():
        raise FileExistsError("Calibrated output is immutable")
    run(
        "calibration",
        "scripts/fit_recovery_policy.py",
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
        "--policy-design",
        "configs/workflow-recovery-policy-v13.json",
    )
    calibration = read(root / "calibration/evaluation.json")
    if not all(fit["passed"] is True for fit in calibration["fitting"].values()):
        stop("calibration", "Fresh calibration fails the unchanged per-type error/coverage gates")
    run(
        "policy-freeze",
        "scripts/freeze_recovery_policy.py",
        "--checkpoint",
        checkpoint,
        "--selection",
        selection_path,
        "--calibration-report",
        root / "calibration/evaluation.json",
        "--output",
        root / "policy-freeze.json",
    )
    for role, spec in selection["independent_policy_validation_records"].items():
        run(
            role,
            "scripts/evaluate_workflow.py",
            "--checkpoint",
            checkpoint,
            "--records",
            spec["path"],
            "--split",
            "calibration",
            "--output",
            root / role,
            "--selection",
            selection_path,
        )
        run(
            role + "-policy-audit",
            "scripts/audit_recovery_policy_population.py",
            "--kind",
            role,
            "--checkpoint",
            checkpoint,
            "--selection",
            selection_path,
            "--calibration-report",
            root / "calibration/evaluation.json",
            "--policy-freeze",
            root / "policy-freeze.json",
            "--evaluation",
            root / role,
            "--output",
            root / (role + "-policy.json"),
        )
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
        "scripts/audit_recovery_policy_population.py",
        "--kind",
        "previous",
        "--checkpoint",
        checkpoint,
        "--selection",
        selection_path,
        "--calibration-report",
        root / "calibration/evaluation.json",
        "--policy-freeze",
        root / "policy-freeze.json",
        "--evaluation",
        root / "previous-calibration",
        "--output",
        root / "previous-calibration-policy.json",
    )
    evidence = collect_recovery_evidence(
        checkpoint,
        selection_path,
        selection["fresh_final_records"],
        root / "calibration/evaluation.json",
        root / "previous-calibration-policy.json",
    )
    write(root / "pre-final-policy-checks.json", evidence)
    record(
        "policy-validation",
        "complete",
        all_policy_checks_passed=True,
        remaining="Final freeze, independent evaluation, regressions and HF preparation",
    )


if __name__ == "__main__":
    root = Path("runs/backbone-recovery-release-v13")
    root.mkdir(parents=True, exist_ok=False)
    try:
        main(root)
    except Exception as error:
        progress_path = root / "progress.json"
        progress = read(progress_path) if progress_path.exists() else {}
        write(
            progress_path,
            {
                **progress,
                "status": "failed",
                "release_allowed": False,
                "error": f"{type(error).__name__}: {error}",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
