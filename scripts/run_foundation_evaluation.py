"""Complete the predeclared GPU evaluation sequence after common-backbone training."""

import json
import subprocess
import sys
from pathlib import Path


def run(script, *arguments):
    command = [sys.executable, "scripts/" + script, *map(str, arguments)]
    print(json.dumps({"running": command}), flush=True)
    subprocess.run(command, check=True)


def main():
    if not Path("runs/foundation-v11-backbone/complete.json").exists():
        raise RuntimeError("Backbone training must finish first")
    root = Path("runs/foundation-v11-evaluation")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "evaluation sequence is immutable; resume individual completed stages explicitly"
        )
    root.mkdir(parents=True)
    corpus = "data/foundation-v11/records.jsonl"
    run(
        "select_foundation_checkpoint.py",
        "--backbone-run",
        "runs/foundation-v11-backbone",
        "--head-run",
        "runs/foundation-v11-head",
        "--records",
        corpus,
        "--output",
        "runs/foundation-v11-selection",
    )
    selected = json.loads(Path("runs/foundation-v11-selection/selection.json").read_text())
    checkpoint = "checkpoints/veyra-foundation-v11"
    run(
        "evaluate_foundation.py",
        "--checkpoint",
        selected["selected"]["checkpoint"],
        "--records",
        corpus,
        "--split",
        "calibration",
        "--output",
        root / "calibration",
        "--calibrated-output",
        checkpoint,
    )
    for name, candidate in (
        ("baseline-final", "checkpoints/veyra-dynamic-v2"),
        ("foundation-final", checkpoint),
    ):
        run(
            "evaluate_foundation.py",
            "--checkpoint",
            candidate,
            "--records",
            corpus,
            "--split",
            "test",
            "--output",
            root / name,
        )
    run(
        "evaluate_foundation.py",
        "--checkpoint",
        checkpoint,
        "--records",
        "data/foundation-v11-typed-regression/records.jsonl",
        "--split",
        "test",
        "--output",
        root / "typed-regression",
    )
    run(
        "validate_model.py",
        "--checkpoint",
        checkpoint,
        "--records",
        "data/starter-v2/records.jsonl",
        "--output",
        root / "runtime.json",
    )
    run(
        "evaluate_foundation.py",
        "--checkpoint",
        checkpoint,
        "--records",
        "data/policy-v8/records.jsonl",
        "--split",
        "test",
        "--legacy-regression",
        "--output",
        root / "legacy-policy-regression",
    )
    run(
        "benchmark_foundation_http.py",
        "--checkpoint",
        checkpoint,
        "--records",
        corpus,
        "--output",
        root / "photo-http.json",
    )
    # Original questions preserve the reference SDK defaults when criteria are absent.
    # Earlier normalized-input diagnostic reports remain intact.
    official = "runs/laya-comparison-20260930/test.jsonl"
    for index, name in ((0, "base"), (1, "specialist")):
        run(
            "evaluate_laya_reference.py",
            "--reference",
            index,
            "--official",
            official,
            "--output",
            f"runs/foundation-v11-laya/official-original-{name}",
        )
        run(
            "evaluate_laya_reference.py",
            "--reference",
            index,
            "--records",
            corpus,
            "--output",
            f"runs/foundation-v11-laya/foundation-{name}",
        )
    run(
        "evaluate_laya_reference.py",
        "--reference",
        0,
        "--official",
        official,
        "--max-len",
        1024,
        "--output",
        "runs/foundation-v11-laya/official-original-base-full-input",
    )
    (root / "complete.json").write_text(
        json.dumps(
            {"completed": True, "checkpoint": checkpoint, "packaging_remaining": True}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
