"""Refresh the existing unpublished Foundation documentation without changing its model."""

import hashlib
import json
import subprocess
from pathlib import Path

from prepare_workflow_huggingface import portable_roadmap


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    root = Path("dist/huggingface/veyra-foundation-v11")
    metadata_path = root.parent / "veyra-foundation-v11-preparation.json"
    checks_path = root / "checksums.json"
    metadata, checks = read(metadata_path), read(checks_path)
    if (
        metadata["published"] is not False
        or metadata["release_allowed"] is not False
        or metadata["checksums_sha256"] != digest(checks_path)
        or not all(digest(root / name) == expected for name, expected in checks.items())
    ):
        raise ValueError("the existing unpublished preparation changed; inspect before editing")
    protected = {
        name: digest(root / name)
        for name in ("head.safetensors", "manifest.json", "runtime/veyra-0.3.0a1-py3-none-any.whl")
    }
    readiness = read("reports/workflow-v12/release-readiness.json")
    if readiness["release_allowed"] is not False or readiness["required_priorities"] != [1, 2]:
        raise ValueError("Foundation preparation must continue to report the unmet release gates")
    (root / "README.md").write_bytes(Path("MODEL_CARD.md").read_bytes())
    for name in ("ROADMAP.md", "ROADMAP.ko.md"):
        (root / name).write_text(
            portable_roadmap(Path(name).read_text(encoding="utf-8")),
            encoding="utf-8",
            newline="\n",
        )
    (root / "release-readiness.json").write_bytes(
        Path("reports/workflow-v12/release-readiness.json").read_bytes()
    )
    checks = {name: digest(root / name) for name in checks}
    write(checks_path, checks)
    metadata["documentation_commit"] = subprocess.check_output(
        ["git", "-c", "safe.directory=C:/tmp/veyra", "rev-parse", "HEAD"], text=True
    ).strip()
    metadata["checksums_sha256"] = digest(checks_path)
    write(metadata_path, metadata)
    if not all(digest(root / name) == expected for name, expected in protected.items()):
        raise ValueError("protected model/runtime changed")
    if not all(digest(root / name) == expected for name, expected in checks.items()):
        raise ValueError("documentation package checksum verification failed")
    print(
        json.dumps(
            {
                "verified_files": len(checks),
                "documentation_commit": metadata["documentation_commit"],
                "model_and_runtime_unchanged": True,
                "repository_report_links_portable": True,
                "published": False,
                "release_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
