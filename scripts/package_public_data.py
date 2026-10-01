"""Package licensed corpus snapshots without changing original split assignments.

The source workspace is explicit; no credentials, caches or unlisted inputs are read.
"""

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

CORPORA = (
    "starter-v2",
    "policy-v3",
    "policy-v6",
    "policy-v8",
    "foundation-v11",
    "workflow-v12",
    "workflow-v12-cohort",
    "workflow-v12b",
    "workflow-v12c",
    "workflow-v12d",
    "workflow-v12e",
    "workflow-v12f",
    "workflow-v13g",
    "workflow-v13h",
    "workflow-v13-final",
)
LICENSES = {"Apache-2.0", "MIT", "CC-BY-4.0", "CC-BY-SA-4.0"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "corpora").mkdir(parents=True)
    summaries = []
    sources = {}
    for corpus in CORPORA:
        root = source / "data" / corpus
        original = root / "records.jsonl"
        if not original.is_file():
            raise FileNotFoundError(original)
        raw = original.read_text("utf-8").splitlines()
        kept = []
        excluded = []
        images = {}
        groups = {}
        image_splits = {}
        ids = set()
        viewers = {}
        for line in raw:
            if not line.strip():
                continue
            row = json.loads(line)
            license_id = row["source"]["license"]
            if license_id not in LICENSES:
                if row["source"]["id"] != "uoft-cs/cifar10" or row["split"] == "train":
                    raise ValueError("Unexpected excluded source or training exclusion")
                excluded.append(
                    {
                        "id": row["id"],
                        "split": row["split"],
                        "source": row["source"],
                        "image_sha256": row["image_sha256"],
                    }
                )
                continue
            if row["id"] in ids:
                raise ValueError("duplicate row ID")
            ids.add(row["id"])
            split = row["split"]
            if groups.setdefault(row["group_id"], split) != split:
                raise ValueError("group leakage within corpus")
            if row["image_sha256"] and image_splits.setdefault(row["image_sha256"], split) != split:
                raise ValueError("image byte leakage within corpus")
            for item in row["request"]["state"].get("images", []):
                asset = (root / item["path"]).resolve()
                if not asset.is_relative_to(root.resolve()):
                    raise ValueError("image outside corpus")
                if asset not in images:
                    actual = digest(asset)
                    if actual != row["image_sha256"]:
                        raise ValueError("image checksum mismatch")
                    images[asset] = {
                        "path": item["path"],
                        "sha256": actual,
                        "bytes": asset.stat().st_size,
                    }
                if images[asset]["sha256"] != row["image_sha256"]:
                    raise ValueError("Repeated image has a conflicting record checksum")
            key = json.dumps(row["source"], sort_keys=True)
            sources[key] = row["source"]
            kept.append(line)
            view = {
                "id": row["id"],
                "group_id": row["group_id"],
                "split": split,
                "family": row["family"],
                "language": row["language"],
                "request_json": json.dumps(row["request"], ensure_ascii=False),
                "targets_json": json.dumps(row["targets"], ensure_ascii=False),
                "source_id": row["source"]["id"],
                "source_revision": row["source"]["revision"],
                "source_license": license_id,
                "source_url": row["source"]["url"],
                "image_sha256": row["image_sha256"],
                "image_paths_json": json.dumps(
                    [i["path"] for i in row["request"]["state"].get("images", [])]
                ),
            }
            viewers.setdefault(split, []).append(view)
        records = ("\n".join(kept) + "\n").encode("utf-8")
        manifest = {
            "corpus": corpus,
            "source_records_sha256": digest(original),
            "published_records_sha256": hashlib.sha256(records).hexdigest(),
            "source_rows": len(raw),
            "published_rows": len(kept),
            "excluded_rows": len(excluded),
            "split_counts": {k: len(v) for k, v in viewers.items()},
            "groups": len(groups),
            "languages": dict(Counter(json.loads(x)["language"] for x in kept)),
            "source_counts": dict(Counter(json.loads(x)["source"]["id"] for x in kept)),
            "images": list(images.values()),
            "within_corpus_group_and_image_split_checks": True,
            "split_assignments_modified": False,
        }
        archive_name = f"qwen3.5-classification-{corpus}.zip"
        zip_path = output / "corpora" / archive_name
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            project = Path(__file__).resolve().parents[1]
            for notice in sorted((project / "docs/data-licenses").glob("*.txt")):
                archive.write(notice, f"{corpus}/licenses/{notice.name}")
            archive.write(project / "LICENSE", f"{corpus}/licenses/Apache-2.0.txt")
            archive.write(project / "NOTICE", f"{corpus}/NOTICE")
            archive.write(project / "docs/DATA.md", f"{corpus}/DATA-LICENSES.md")
            archive.writestr(f"{corpus}/records.jsonl", records)
            archive.writestr(
                f"{corpus}/publication-manifest.json", json.dumps(manifest, indent=2) + "\n"
            )
            for asset, info in sorted(images.items()):
                archive.write(asset, f"{corpus}/" + info["path"])
        write(output / "manifests" / f"{corpus}.json", manifest)
        write(output / "excluded" / f"{corpus}.json", excluded)
        for split, rows in viewers.items():
            target = output / "viewer" / corpus / f"{split}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8"
            )
        summary = {k: v for k, v in manifest.items() if k != "images"}
        summary.update(
            archive=f"corpora/{archive_name}",
            archive_sha256=digest(zip_path),
            archive_bytes=zip_path.stat().st_size,
            image_files=len(images),
        )
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "corpus": corpus,
                    "rows": len(kept),
                    "excluded": len(excluded),
                    "images": len(images),
                    "archive_mb": round(zip_path.stat().st_size / 1e6, 2),
                }
            ),
            flush=True,
        )
    write(
        output / "dataset-manifest.json",
        {
            "format_version": 1,
            "project": "qwen3.5-classification",
            "corpora": summaries,
            "sources": list(sources.values()),
            "licenses_per_record": True,
            "cross_corpus_warning": (
                "Configurations are historical stages with overlapping observations. "
                "Do not concatenate or claim disjointness across configurations."
            ),
            "excluded_source": (
                "CIFAR-10 image redistribution license unresolved; evaluation-only "
                "records/images omitted. No selected training rows excluded."
            ),
            "generator_source_sha256": digest(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
