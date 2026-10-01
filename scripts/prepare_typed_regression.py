"""Reproduce the frozen official text benchmark inputs without training on them."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "d51d993547ad8355b1c25157fbc1fea0649e8ffa"
PARQUET_SHA = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
JSONL_SHA = "4881baae1cfd752311a58064cb2095c0457ee2c0b8914d1cfe2590e19933e7b3"
RECORDS_SHA = "f9eaf3ec7848cc4a0e05f3fa2ae210aa8be65a80b7698f3d55c57dd541725b35"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def preserve(path, blob, expected):
    if digest(blob) != expected:
        raise ValueError("unexpected source bytes: " + str(path))
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError("refusing to replace different existing data: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-jsonl-only", action="store_true")
    args = parser.parse_args()
    root = Path("runs/laya-comparison-20260930")
    parquet, jsonl = root / "test.parquet", root / "test.jsonl"
    if args.download_jsonl_only:
        if parquet.exists():
            blob = parquet.read_bytes()
        else:
            base = "https://huggingface.co"
            with urllib.request.urlopen(
                f"{base}/api/datasets/LocalLLaMA/typed-decisions/tree/{REVISION}/all", timeout=60
            ) as response:
                entries = json.load(response)
            paths = [p["path"] for p in entries if p["path"].startswith("all/test-")]
            if len(paths) != 1 or not paths[0].endswith(".parquet"):
                raise ValueError("unexpected pinned dataset layout")
            url = f"{base}/datasets/LocalLLaMA/typed-decisions/resolve/{REVISION}/{paths[0]}"
            with urllib.request.urlopen(url, timeout=60) as response:
                blob = response.read()
        preserve(parquet, blob, PARQUET_SHA)
        import pyarrow.parquet as pq

        rows = pq.read_table(parquet).to_pylist()
        # Preserve the original Windows experiment's byte hashes on every platform.
        text = "".join(json.dumps(row, ensure_ascii=False) + "\r\n" for row in rows)
        preserve(jsonl, text.encode("utf-8"), JSONL_SHA)
        print(json.dumps({"cases": len(rows), "sha256": JSONL_SHA}))
        return
    if digest(jsonl.read_bytes()) != JSONL_SHA:
        raise ValueError("run the pinned Parquet-to-JSONL preparation first")
    from evaluate_laya_reference import official_records

    records = official_records(jsonl)
    blob = "".join(row.model_dump_json() + "\r\n" for row in records).encode("utf-8")
    output = Path("data/foundation-v11-typed-regression/records.jsonl")
    preserve(output, blob, RECORDS_SHA)
    metadata = {
        "dataset": "LocalLLaMA/typed-decisions",
        "revision": REVISION,
        "parquet_sha256": PARQUET_SHA,
        "original_jsonl_sha256": JSONL_SHA,
        "records_sha256": RECORDS_SHA,
        "questions": len(records),
        "groups": len({row.group_id for row in records}),
        "scope": "Previously inspected regression test, excluded from all foundation training",
        "source_sha256": digest(Path(__file__).read_bytes()),
        "adapter_sha256": digest(Path("scripts/evaluate_laya_reference.py").read_bytes()),
        "byte_reproduction": "Explicit CRLF to preserve the original experiment hashes",
    }
    (output.parent / "protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"questions": len(records), "sha256": RECORDS_SHA}))


if __name__ == "__main__":
    main()
