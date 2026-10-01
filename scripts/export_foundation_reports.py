"""Copy aggregate evidence and frozen protocols into the public source repository."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/foundation-v11"))
    args = parser.parse_args()
    output = args.output
    paths = {
        "data-protocol.json": Path("data/foundation-v11/protocol.json"),
        "source-downloads.json": Path("data/foundation-v11-source/downloads.json"),
        "head-protocol.json": Path("runs/foundation-v11-head/protocol.json"),
        "head-comparison.json": Path("runs/foundation-v11-head/results.json"),
        "head-baseline.json": Path("runs/foundation-v11-head/baseline.json"),
        "head-selection.json": Path("runs/foundation-v11-head/selection.json"),
        "head-training-fit.json": Path("runs/foundation-v11-head/training-fit.json"),
        "backbone-protocol.json": Path("runs/foundation-v11-backbone/protocol.json"),
        "backbone-history.json": Path("runs/foundation-v11-backbone/history.json"),
        "midpoint-amendment.json": Path("runs/foundation-v11-backbone/midpoint-amendment.json"),
        "selection.json": Path("runs/foundation-v11-selection/selection.json"),
        "selection-protocol.json": Path("runs/foundation-v11-selection/protocol.json"),
        "laya-sources.json": Path("runs/foundation-v11-laya/sources.json"),
        "typed-data-protocol.json": Path("data/foundation-v11-typed-regression/protocol.json"),
    }
    for name in (
        "calibration",
        "baseline-final",
        "foundation-final",
        "typed-regression",
        "legacy-policy-regression",
    ):
        paths[name + ".json"] = Path("runs/foundation-v11-evaluation") / name / "evaluation.json"
    for index in range(4):
        paths[f"merged-dev-{index}.json"] = (
            Path("runs/foundation-v11-selection") / f"candidate-{index}/evaluation.json"
        )
    for name in ("runtime", "photo-http", "paired-comparison"):
        paths[name + ".json"] = Path("runs/foundation-v11-evaluation") / (name + ".json")
    for name in ("typed-regression-audit", "typed-baseline-standardized"):
        audit = Path("runs/foundation-v11-evaluation") / (name + ".json")
        if audit.exists():
            paths[name + ".json"] = audit
    for name in (
        "official-original-base",
        "official-original-specialist",
        "foundation-base",
        "foundation-specialist",
        "official-original-base-full-input",
    ):
        paths["laya-" + name + ".json"] = (
            Path("runs/foundation-v11-laya") / name / "evaluation.json"
        )
    for name in ("official-base", "official-specialist"):
        diagnostic = Path("runs/foundation-v11-laya") / name / "evaluation.json"
        if diagnostic.exists():
            paths["laya-" + name + ".json"] = diagnostic
    for name in (
        "original-base-input-audit",
        "original-specialist-input-audit",
        "original-base-full-input-audit",
        "foundation-base-input-audit",
    ):
        paths["laya-" + name + ".json"] = Path("runs/foundation-v11-laya") / (name + ".json")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Unfinished reports: " + ", ".join(missing))
    output.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, path in paths.items():
        target = output / name
        if target.exists():
            if digest(path) != digest(target):
                raise ValueError("refusing to replace different public evidence: " + name)
        else:
            shutil.copyfile(path, target)
        entries[name] = {"source": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
    manifest = {
        "scope": "Aggregate evaluations and protocols; excludes raw texts, images and credentials",
        "source_sha256": digest(Path(__file__)),
        "entries": entries,
    }
    (output / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"exported": len(entries), "folder": str(output)}))


if __name__ == "__main__":
    main()
