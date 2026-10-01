"""Rename a verified dataset release while preserving every corpus archive byte."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.prepared.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Use a new, separate dataset directory")
    checks = json.loads((source / "checksums.json").read_text("utf-8"))
    for name, expected in checks.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha(path) != expected:
            raise ValueError("Prepared dataset changed: " + name)
    for name in checks:
        destination = name.replace("corpora/qwen3.5-classification-", "corpora/qev-")
        path = output / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, path)
    manifest = json.loads((output / "dataset-manifest.json").read_text("utf-8"))
    manifest["project"] = "qev"
    for corpus in manifest["corpora"]:
        corpus["archive"] = corpus["archive"].replace("qwen3.5-classification-", "qev-")
        if sha(output / corpus["archive"]) != corpus["archive_sha256"]:
            raise ValueError("Corpus archive bytes changed")
    (output / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    print(json.dumps({"corpora": len(manifest["corpora"]), "archive_bytes_preserved": True}))


if __name__ == "__main__":
    main()
