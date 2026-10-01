"""Copy completed aggregate follow-up evidence without raw records or model outputs."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = Path("reports/workflow-v12/policy-calibration")
    root = Path("runs/workflow-v12-depth-policy-release")
    selection = Path("runs/workflow-v12-depth-policy-selection")
    pairs = {
        "population-shift-diagnosis.json": root / "population-shift-diagnosis.json",
        "release-status.json": root / "progress.json",
        "selection.json": selection / "selection.json",
        "data-and-lineage-audit.json": selection / "data-and-lineage-audit.json",
        "calibration.json": root / "calibration/evaluation.json",
        "policy-freeze.json": root / "policy-freeze.json",
        "validation-evaluation.json": root / "policy-validation/evaluation.json",
        "independent-policy-validation.json": root / "independent-policy-validation.json",
        "previous-evaluation.json": root / "previous-calibration/evaluation.json",
        "previous-calibration-policy.json": root / "previous-calibration-policy.json",
        "runtime-version.json": root / "runtime-version.json",
        "runtime-wheel.json": root / "runtime-wheel.json",
        "final-freeze.json": root / "final-freeze.json",
        "baseline-final.json": root / "baseline/evaluation.json",
        "candidate-final.json": root / "final/evaluation.json",
        "paired-final.json": root / "paired.json",
        "official-regression.json": root / "official-regression/evaluation.json",
        "legacy-regression.json": root / "legacy-regression/evaluation.json",
        "foundation-regression.json": root / "foundation-regression/evaluation.json",
        "runtime.json": root / "runtime.json",
        "photo-http.json": root / "photo-http.json",
        "network-contract.json": root / "network-contract.json",
        "release-acceptance.json": root / "release-acceptance.json",
        "hf-package-verification.json": root / "hf-package-verification.json",
    }
    copied = []
    for name, source in pairs.items():
        if not source.is_file():
            continue
        # Reject partial JSON; never export records, predictions, logits or a checkpoint.
        json.loads(source.read_text(encoding="utf-8"))
        target = output / name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError("Immutable public evidence changed: " + name)
        else:
            target.write_bytes(source.read_bytes())
            copied.append(name)
    manifest = {
        path.name: digest(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"README.md", "manifest.json"}
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"exported": copied, "manifest_files": len(manifest)}))


if __name__ == "__main__":
    main()
