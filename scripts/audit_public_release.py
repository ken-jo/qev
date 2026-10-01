"""Verify the publishable source, model and licensed dataset payloads."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

PRIVATE_MARKDOWN = re.compile("[\uac00-\ud7a3]")
SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{25,}|ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_package(folder):
    expected = json.loads((folder / "checksums.json").read_text("utf-8"))
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if actual != set(expected) | {"checksums.json"}:
        raise ValueError("Package has unlisted or missing files")
    for name, sha in expected.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path.read_bytes()) != sha:
            raise ValueError("Package checksum mismatch: " + name)
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    counts = {"model": verify_package(args.model), "dataset": verify_package(args.data)}
    corpora = json.loads((args.data / "dataset-manifest.json").read_text("utf-8"))["corpora"]
    for corpus in corpora:
        archive_path = args.data / corpus["archive"]
        if digest(archive_path.read_bytes()) != corpus["archive_sha256"]:
            raise ValueError("Dataset archive mismatch")
        prefix = corpus["corpus"] + "/"
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Corrupt dataset ZIP")
            if not all(
                n.startswith(prefix) and ".." not in Path(n).parts for n in archive.namelist()
            ):
                raise ValueError("Unsafe dataset ZIP path")
            if prefix + "licenses/Apache-2.0.txt" not in archive.namelist():
                raise ValueError("Standalone dataset archive lacks notices")
            raw = archive.read(prefix + "records.jsonl")
            if digest(raw) != corpus["published_records_sha256"]:
                raise ValueError("Published dataset record mismatch")
            records = [json.loads(line) for line in raw.decode().splitlines() if line]
            if len(records) != corpus["published_rows"]:
                raise ValueError("Dataset count mismatch")
            manifest = json.loads(archive.read(prefix + "publication-manifest.json"))
            for image in manifest["images"]:
                if digest(archive.read(prefix + image["path"])) != image["sha256"]:
                    raise ValueError("Packaged image checksum mismatch")
            for record in records:
                if record["source"]["license"] not in {
                    "Apache-2.0",
                    "MIT",
                    "CC-BY-4.0",
                    "CC-BY-SA-4.0",
                }:
                    raise ValueError("Uncleared source found in dataset")
    documents = []
    inspected = 0
    for folder in (root, args.model, args.data):
        for path in folder.rglob("*"):
            rel = path.relative_to(folder)
            if folder == root and rel.parts[0] in {"dist", "runs", ".cache", ".git", ".venv"}:
                continue
            if not path.is_file() or path.suffix not in {
                ".md",
                ".json",
                ".jsonl",
                ".py",
                ".html",
                ".js",
                ".ps1",
                ".toml",
                ".txt",
            }:
                continue
            text = path.read_text("utf-8", errors="replace")
            inspected += 1
            if SECRET.search(text):
                raise ValueError("Credential-like string found; review privately: " + str(rel))
            if path.suffix == ".md":
                if PRIVATE_MARKDOWN.search(text) or path.name.endswith(".ko.md"):
                    raise ValueError("Private-language document in public payload: " + str(rel))
                documents.append(rel.as_posix())
    active = [
        root / p
        for p in (
            "README.md",
            "MODEL_CARD.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "reports/README.md",
            "apps/playground/README.md",
        )
    ]
    active += list((root / "docs").glob("*.md"))
    for path in active:
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text("utf-8")):
            if "://" in link or link.startswith("#"):
                continue
            if not (path.parent / link.split("#")[0]).exists():
                raise ValueError("Broken active documentation link: " + str(path) + ": " + link)
    inference = json.loads((root / "release/model-inference.json").read_text("utf-8"))
    if (
        inference.get("passed") is not True
        or inference.get("qev_version") != "0.1.1"
    ):
        raise ValueError("Real packaged model inference has not passed")
    if inference.get("checksums_sha256") != digest((args.model / "checksums.json").read_bytes()):
        raise ValueError("Inference report belongs to a different model package")
    report = {
        "passed": True,
        "project": "qev",
        "version": "0.1.1",
        "package_files": counts,
        "dataset_configurations": len(corpora),
        "dataset_records_including_cross_stage_repetition": sum(
            c["published_rows"] for c in corpora
        ),
        "dataset_archive_bytes": sum(c["archive_bytes"] for c in corpora),
        "all_packaged_image_hashes_verified": True,
        "all_archives_include_license_notices": True,
        "source_license_allowlist_passed": True,
        "public_markdown_files_checked": len(documents),
        "private_korean_documents_in_public_payload": 0,
        "text_files_scanned": inspected,
        "credential_pattern_scan_passed": True,
        "active_documentation_links_passed": True,
        "real_packaged_text_and_three_type_image_inference_passed": True,
        "source_sha256": digest(Path(__file__).read_bytes()),
        "model_checksums_sha256": digest((args.model / "checksums.json").read_bytes()),
        "data_checksums_sha256": digest((args.data / "checksums.json").read_bytes()),
        "scope": "Publication package audit; not a new accuracy study or clean-machine install",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
