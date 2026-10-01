"""Download pinned public sources without credentials; retain byte provenance."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCES = {
    "trashnet-kaggle.zip": "https://www.kaggle.com/api/v1/datasets/download/vminhkhoi/trashnet?datasetVersionNumber=1",
    "trashnet-original.zip": "https://huggingface.co/datasets/garythung/trashnet/resolve/94cd17ebfd6702bf62281d3c89f22ff649815b46/dataset-resized.zip",
    "trashnet-README.md": "https://huggingface.co/datasets/garythung/trashnet/resolve/94cd17ebfd6702bf62281d3c89f22ff649815b46/README.md",
    "trashnet-kaggle-metadata.json": "https://www.kaggle.com/api/v1/datasets/view/vminhkhoi/trashnet",
    "snli-README.md": "https://huggingface.co/datasets/stanfordnlp/snli/resolve/cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/README.md",
    **{
        f"snli-{split}.parquet": (
            "https://huggingface.co/datasets/stanfordnlp/snli/resolve/"
            "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/plain_text/"
            f"{split}-00000-of-00001.parquet"
        )
        for split in ("train", "validation", "test")
    },
    **{
        f"banking-{name}": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/{name}"
        )
        for name in ("train.csv", "test.csv", "categories.json")
    },
    "banking-README.md": "https://huggingface.co/datasets/PolyAI/banking77/resolve/90d4e2ee5521c04fc1488f065b8b083658768c57/README.md",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/foundation-v11-source"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous_path = args.output / "downloads.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}

    def download(item):
        name, url = item
        target = args.output / name
        if target.exists():
            checksum = hashlib.sha256(target.read_bytes()).hexdigest()
            if name not in previous or previous[name]["sha256"] != checksum:
                raise ValueError(f"unverified existing source: {name}")
            return name, previous[name]
        request = urllib.request.Request(url, headers={"User-Agent": "Veyra-research/0.3"})
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                raise ValueError(f"unexpected HTML rather than source data: {name}")
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(target)
        result = {
            "url": url,
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        print(json.dumps({"downloaded": name, **result}), flush=True)
        return name, result

    with ThreadPoolExecutor(max_workers=3) as pool:
        for name, result in pool.map(download, SOURCES.items()):
            previous[name] = result
            previous_path.write_text(
                json.dumps(previous, indent=2) + "\n", encoding="utf-8", newline="\n"
            )


if __name__ == "__main__":
    main()
