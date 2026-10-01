"""Fit, independently validate and release a policy constrained across three cohorts."""

import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(root):
    checkpoint = Path("checkpoints/veyra-cohort-policy-v12")
    data = Path("data/workflow-v12-cohort/records.jsonl")
    selection_path = Path("runs/workflow-v12-cohort-policy-selection/selection.json")

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
    audit_path = selection_path.parent / "data-and-lineage-audit.json"
    audit = read(audit_path)
    if not selection["eligible"] or not audit["passed"]:
        stop("selection", "Fresh-data lineage and unchanged weight selection must pass")
    for path, expected in audit["evidence_files"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            stop("selection", "Data or source changed after preparation: " + path)
    if checkpoint.exists():
        raise FileExistsError("Calibrated output is immutable")
    run(
        "calibration",
        "scripts/fit_cohort_policy.py",
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
        "configs/workflow-cohort-policy-v12.json",
    )
    calibration = read(root / "calibration/evaluation.json")
    if not all(fit["passed"] is True for fit in calibration["fitting"].values()):
        stop("calibration", "Fresh calibration fails the unchanged per-type error/coverage gates")
    run(
        "policy-freeze",
        "scripts/freeze_cohort_policy.py",
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
            "scripts/audit_cohort_policy_population.py",
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
        "scripts/audit_cohort_policy_population.py",
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
    run(
        "runtime-version",
        "scripts/prepare_cohort_policy_runtime.py",
        "--checkpoint",
        checkpoint,
        "--selection",
        selection_path,
        "--records",
        data,
        "--evaluation",
        root,
        "--output",
        root / "runtime-version.json",
    )
    wheel = Path("dist/workflow-runtime/veyra-0.3.0a3-py3-none-any.whl")
    if wheel.exists():
        raise FileExistsError("Versioned runtime wheel is immutable")
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
    with zipfile.ZipFile(wheel) as archive:
        modules = sorted(Path("src/veyra").rglob("*.py"))
        for path in modules:
            if archive.read(path.relative_to("src").as_posix()) != path.read_bytes():
                raise ValueError("Built wheel differs from source: " + str(path))
    write(
        root / "runtime-wheel.json",
        {
            "passed": True,
            "wheel": str(wheel),
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "python_modules_compared": len(modules),
            "version": "0.3.0a3",
            "release_allowed": False,
        },
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
    root = Path("runs/workflow-v12-cohort-policy-release")
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
