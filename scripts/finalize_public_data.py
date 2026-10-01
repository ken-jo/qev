"""Add the public dataset card, license notices and manifest checksums."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = args.package
    manifest = json.loads((data / "dataset-manifest.json").read_text("utf-8"))
    lines = [
        "---",
        "license: other",
        "license_name: source-specific-licenses",
        "license_link: https://github.com/ken-jo/qev/blob/main/docs/DATA.md",
        "language:",
        "- en",
        "- ko",
        "task_categories:",
        "- text-classification",
        "- image-classification",
        "tags:",
        "- multimodal",
        "- typed-decisions",
        "- synthetic",
        "configs:",
    ]
    for corpus in manifest["corpora"]:
        name = corpus["corpus"]
        lines += ["- config_name: " + name, "  data_files:"]
        for split in ("train", "dev", "calibration", "test"):
            if split in corpus["split_counts"]:
                lines += [f"  - split: {split}", f"    path: viewer/{name}/{split}.jsonl"]
    lines += [
        "---",
        "",
        "# QEV data",
        "",
        "Historical training snapshots for QEV, the LAYA-inspired Qwen3.5-2B decision model.",
        "[Model](https://huggingface.co/ken-jo/qev).",
        "",
        "**Configurations overlap. Do not concatenate them or assume independent test sets.**",
        "",
        "ZIPs in `corpora/` use `qev-<stage>.zip` and contain eligible",
        "original records, images and license notices. Extracted stage folders and original",
        "record/source IDs preserve the recorded training provenance.",
        "Viewer rows expose request/target schemas as JSON strings; parse with `json.loads`.",
        "IDs, group IDs, language and original split assignments are preserved.",
        "",
        "| Configuration | Rows | Train | Dev | Calibration | Test |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for corpus in manifest["corpora"]:
        counts = " | ".join(
            str(corpus["split_counts"].get(s, 0)) for s in ("train", "dev", "calibration", "test")
        )
        lines.append(f"| {corpus['corpus']} | {corpus['published_rows']} | {counts} |")
    lines += [
        "",
        "Counts include repeated observations across stages; the sum is not a unique count.",
        "",
        "## Licenses and exclusions",
        "",
        "This collection is not uniformly Apache-2.0. Read [LICENSES.md](LICENSES.md),",
        "per-record source/license fields, and the original notices in `licenses/`.",
        "Authored data and typed-decisions use Apache-2.0; Beans/TrashNet use MIT;",
        "BANKING77 uses CC-BY-4.0; SNLI derivatives retain CC-BY-SA-4.0.",
        "Original observations were selected and converted into typed decision views.",
        "Synthetic labels are not observed outcomes; typed-decisions uses teacher agreement.",
        "",
        "CIFAR-10 evaluation requests/images are excluded because redistribution rights are",
        "unresolved. `excluded/` records references and hashes for direct upstream reconstruction.",
        "No selected training rows were removed by that exclusion.",
        "",
        "## Use and limitations",
        "",
        "Extract one corpus ZIP under `data/` in the source repository. Image paths are relative",
        "to that corpus. Manifests record original and exported record/image hashes.",
        "Group and image-byte split checks apply within each corpus, not across historical stages.",
        "They do not prove distinct physical objects or absence from Qwen pretraining.",
        "Korean generated examples are retained; personal project notes are not included.",
        "Old test groups have been inspected. Future models need untouched final populations.",
        "Full retraining from all historical intermediate checkpoints is not turnkey.",
        "",
        "[Training details](https://github.com/ken-jo/qev/blob/main/docs/TRAINING.md).",
        "",
        "[GitHub: ken-jo/qev](https://github.com/ken-jo/qev)",
        "",
    ]
    (data / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    shutil.copytree(root / "docs/data-licenses", data / "licenses", dirs_exist_ok=True)
    shutil.copy2(root / "LICENSE", data / "licenses/Apache-2.0.txt")
    shutil.copy2(root / "NOTICE", data / "NOTICE")
    (data / "LICENSES.md").write_text(
        (root / "docs/DATA.md").read_text("utf-8")
        + "\n## Original notices\n\nSee `licenses/`; each corpus ZIP includes these notices.\n",
        encoding="utf-8",
    )
    (root / "release").mkdir(exist_ok=True)
    shutil.copy2(data / "dataset-manifest.json", root / "release/dataset-manifest.json")
    checks = {
        p.relative_to(data).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(data.rglob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (data / "checksums.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"configs": len(manifest["corpora"]), "files": len(checks) + 1}))


if __name__ == "__main__":
    main()
