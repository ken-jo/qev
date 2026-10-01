"""Finish the depth release evaluation and HF preparation after eligible selection."""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(root):
    checkpoint = Path("checkpoints/veyra-depth-v12")
    data = Path("data/workflow-v12b/records.jsonl")
    selection_path = Path("runs/workflow-v12-depth-selection/selection.json")

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

    record("selection", "waiting")
    while True:
        try:
            progress = read("runs/workflow-v12-depth-evaluation/progress.json")
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(10)
            continue
        if progress["status"] in {"failed", "gate_failed"}:
            stop("selection", "Training or merged selection did not qualify; inspect its evidence")
        if progress["stage"] == "selection" and progress["status"] == "complete":
            break
        time.sleep(10)
    selection = read(selection_path)
    if selection["eligible"] is not True:
        stop(
            "selection",
            "No declared candidate passes development and per-type abstention requirements",
        )
    run(
        "development-audit",
        "scripts/audit_depth_development.py",
        "--selection",
        selection_path,
        "--records",
        data,
        "--output",
        root / "development-audit.json",
    )
    run(
        "runtime-version",
        "scripts/prepare_workflow_runtime_version.py",
        "--selection",
        selection_path,
        "--output",
        root / "runtime-version.json",
    )
    run(
        "runtime-wheel",
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        "dist/workflow-runtime",
        "--cache-dir",
        ".cache/uv",
        "--python",
        sys.executable,
        "--no-python-downloads",
        "--no-create-gitignore",
        python=False,
    )
    wheel = Path("dist/workflow-runtime/veyra-0.3.0a2-py3-none-any.whl")
    if not wheel.is_file():
        raise FileNotFoundError("the declared runtime wheel was not built")
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
        "--study-config",
        "configs/workflow-depth-study-v12.json",
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
    final_started = True
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
    package = Path("dist/huggingface/veyra-workflow-v12")
    run(
        "hf-preparation",
        "scripts/prepare_workflow_huggingface.py",
        "--checkpoint",
        checkpoint,
        "--evaluation",
        root,
        "--wheel",
        wheel,
        "--output",
        package,
    )
    run(
        "hf-verification",
        "-I",
        "scripts/verify_prepared_release.py",
        "--package",
        package,
        "--records",
        "data/foundation-v11/records.jsonl",
        "--output",
        root / "hf-package-verification.json",
    )
    record(
        "evaluation-and-hf-preparation",
        "complete",
        release_allowed=True,
        published=False,
        hf_package=str(package),
        verification_report=str(root / "hf-package-verification.json"),
        verification_report_sha256=hashlib.sha256(
            (root / "hf-package-verification.json").read_bytes()
        ).hexdigest(),
        remaining="Actual HF publication awaits a separately configured account and target",
    )


if __name__ == "__main__":
    root = Path("runs/workflow-v12-depth-release")
    root.mkdir(parents=True, exist_ok=False)
    try:
        main(root)
    except Exception as error:
        progress_path = root / "progress.json"
        if progress_path.exists():
            progress = read(progress_path)
            # Preserve the last stage and whether final inference was already started.
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
