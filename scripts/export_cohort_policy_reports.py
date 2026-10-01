"""Export completed cohort-policy aggregates without copying data, logits or weights."""

import hashlib
import json
from pathlib import Path


def main():
    output = Path("reports/workflow-v12/cohort-policy")
    output.mkdir(parents=True, exist_ok=True)
    root = Path("runs/workflow-v12-cohort-policy-release")
    selection = Path("runs/workflow-v12-cohort-policy-selection")
    pairs = {
        "selection.json": selection / "selection.json",
        "data-and-lineage-audit.json": selection / "data-and-lineage-audit.json",
        "combined-data-protocol.json": Path("data/workflow-v12-cohort/protocol.json"),
        "combined-data-failure.json": Path(
            "runs/workflow-v12-cohort-policy-data/combined-data-failure.json"
        ),
        "validation-data-complete.json": Path("runs/workflow-v12-cohort-policy-data/complete.json"),
        "calibration.json": root / "calibration/evaluation.json",
        "policy-freeze.json": root / "policy-freeze.json",
        "previous-evaluation.json": root / "previous-calibration/evaluation.json",
        "previous-calibration-policy.json": root / "previous-calibration-policy.json",
        "runtime-version.json": root / "runtime-version.json",
        "runtime-wheel.json": root / "runtime-wheel.json",
        "final-freeze.json": root / "final-freeze.json",
        "baseline-final.json": root / "baseline/evaluation.json",
        "candidate-final.json": root / "final/evaluation.json",
        "paired-final.json": root / "paired.json",
    }
    for role, folder in (("validation-1", "workflow-v12e"), ("validation-2", "workflow-v12f")):
        pairs.update(
            {
                role + "-data-protocol.json": Path("data") / folder / "protocol.json",
                role + "-target-audit.json": Path("data")
                / folder
                / "independent-target-audit.json",
                role + "-evaluation.json": root / role / "evaluation.json",
                role + "-policy.json": root / (role + "-policy.json"),
            }
        )
    for name in ("official-regression", "legacy-regression", "foundation-regression"):
        pairs[name + ".json"] = root / name / "evaluation.json"
    for name in (
        "runtime",
        "photo-http",
        "network-contract",
        "release-acceptance",
        "hf-package-verification",
    ):
        pairs[name + ".json"] = root / (name + ".json")
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    if progress["status"] in {"failed", "gate_failed"} or (
        progress["status"] == "complete" and progress["stage"] == "evaluation-and-hf-preparation"
    ):
        pairs["release-status.json"] = root / "progress.json"
    copied = []
    for name, source in pairs.items():
        if not source.is_file():
            continue
        content = source.read_bytes()
        json.loads(content)
        target = output / name
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError("Immutable public aggregate changed: " + name)
        else:
            target.write_bytes(content)
            copied.append(name)
    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"README.md", "manifest.json"}
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"exported": copied, "manifest_files": len(manifest)}))


if __name__ == "__main__":
    main()
