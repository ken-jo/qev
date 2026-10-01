"""Pin and fetch the small CIFAR evaluation shard; leave original data local."""

import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    root = Path("data/workflow-v12-source")
    root.mkdir(parents=True, exist_ok=True)
    output = root / "downloads.json"
    if output.exists():
        raise FileExistsError("source selection is already frozen")
    repo = "uoft-cs/cifar10"
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=30) as r:
        metadata = json.load(r)
    revision = metadata["sha"]
    files = [s["rfilename"] for s in metadata["siblings"]]
    shard = next(n for n in files if n.startswith("plain_text/test-") and n.endswith(".parquet"))
    result = {"dataset": repo, "revision": revision, "license": "unknown", "evaluation_only": True}
    result["files"] = {}
    for upstream, local in ((shard, "cifar-test.parquet"), ("README.md", "cifar-README.md")):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{upstream}"
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read()
        (root / local).write_bytes(raw)
        result["files"][local] = {
            "url": url,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
