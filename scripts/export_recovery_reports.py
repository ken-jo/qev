"""Publish completed recovery aggregates; keep all source observations and weights local."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(folder, pairs):
    folder.mkdir(parents=True, exist_ok=True)
    for name, source in pairs.items():
        source = Path(source)
        if not source.is_file():
            continue
        content = source.read_bytes()
        json.loads(content)
        target = folder / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError("Existing public aggregate differs: " + str(target))
        if not target.exists():
            target.write_bytes(content)
    manifest = {
        p.name: digest(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name not in {"README.md", "manifest.json"}
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return len(manifest)


def main():
    root = Path("reports/workflow-v13")
    study = Path("runs/readout-recovery-v13")
    pairs = {
        name: study / name
        for name in (
            "interpolation-protocol.json",
            "interpolation-complete.json",
            "training-protocol.json",
            "training-complete.json",
            "selection.json",
        )
    }
    pairs.update(
        {
            "feature-specification.json": "data/features-readout-recovery-v13/specification.json",
            "feature-manifest.json": "data/features-readout-recovery-v13/manifest.json",
        }
    )
    counts = {"readout-recovery": export(root / "readout-recovery", pairs)}
    pairs = {"complete.json": "runs/workflow-recovery-final-data-v13/complete.json"}
    for role, folder in (
        ("validation-1", "workflow-v13g"),
        ("validation-2", "workflow-v13h"),
        ("final", "workflow-v13-final"),
    ):
        for suffix, filename in (
            ("protocol", "protocol.json"),
            ("target-audit", "independent-target-audit.json"),
        ):
            pairs[role + "-" + suffix + ".json"] = Path("data") / folder / filename
    for name, folder in (
        ("initial-preparation", "workflow-recovery-data-v13"),
        ("source-pool-repair", "workflow-recovery-data-repaired-v13"),
    ):
        pairs[name + "-execution.json"] = Path("runs") / folder / "execution-design.json"
        pairs[name + "-failure.json"] = Path("runs") / folder / "progress.json"
    pairs["final-count-repair-execution.json"] = (
        "runs/workflow-recovery-final-data-v13/execution.json"
    )
    counts["fresh-data"] = export(root / "fresh-data", pairs)
    backbone = Path("runs/backbone-recovery-v13")
    pairs = {
        name: backbone / name
        for name in (
            "protocol.json",
            "parameter-plan.json",
            "training-complete.json",
            "selection.json",
            "execution-failure.json",
        )
    }
    for fraction in ("0.25", "0.5", "1"):
        pairs["fraction-" + fraction + "-development.json"] = (
            backbone / ("fraction-" + fraction) / "development.json"
        )
    counts["backbone-recovery"] = export(root / "backbone-recovery", pairs)
    retry = Path("runs/backbone-recovery-memory-v13")
    pairs = {
        name: retry / name
        for name in (
            "protocol.json",
            "parameter-plan.json",
            "training-complete.json",
            "selection.json",
            "execution-failure.json",
            "failed-training-row.json",
            "passed-prior-failure.json",
        )
    }
    pairs["training-memory-probe.json"] = "runs/backbone-recovery-memory-probe-v13.json"
    for fraction in ("0.25", "0.5", "1"):
        pairs["fraction-" + fraction + "-development.json"] = (
            retry / ("fraction-" + fraction) / "development.json"
        )
        if fraction != "1":
            pairs["fraction-" + fraction + "-execution-equivalence.json"] = (
                retry / ("fraction-" + fraction) / "execution-equivalence.json"
            )
    counts["backbone-memory-repair"] = export(root / "backbone-memory-repair", pairs)
    release = Path("runs/backbone-recovery-release-v13")
    admission = Path("runs/backbone-recovery-policy-selection-v13")
    pairs = {
        "selection.json": admission / "selection.json",
        "admission-audit.json": admission / "admission-audit.json",
        "calibration.json": release / "calibration/evaluation.json",
        "policy-freeze.json": release / "policy-freeze.json",
        "previous-evaluation.json": release / "previous-calibration/evaluation.json",
        "previous-calibration-policy.json": release / "previous-calibration-policy.json",
        "pre-final-policy-checks.json": release / "pre-final-policy-checks.json",
    }
    for role in ("validation-1", "validation-2"):
        pairs[role + "-evaluation.json"] = release / role / "evaluation.json"
        pairs[role + "-policy.json"] = release / (role + "-policy.json")
    if any(path.is_file() for path in pairs.values()):
        counts["policy-validation"] = export(root / "policy-validation", pairs)
    pairs = {
        "baseline-final.json": release / "baseline/evaluation.json",
        "candidate-final.json": release / "final/evaluation.json",
        "paired-final.json": release / "paired.json",
    }
    for name in ("official-regression", "legacy-regression", "foundation-regression"):
        pairs[name + ".json"] = release / name / "evaluation.json"
    for name in (
        "runtime-transition",
        "runtime-wheel",
        "final-freeze",
        "runtime",
        "photo-http",
        "network-contract",
        "release-acceptance",
        "hf-package-verification",
    ):
        pairs[name + ".json"] = release / (name + ".json")
    # Copy only completed, immutable reports. The coordinator's progress.json changes
    # during execution and must never be mistaken for a completed acceptance report.
    if any(path.is_file() for path in pairs.values()):
        counts["final-evaluation"] = export(root / "final-evaluation", pairs)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
