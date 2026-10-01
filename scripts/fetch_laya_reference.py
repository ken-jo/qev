"""Pin and download public Laya reference checkpoints without loading a GPU model."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REFERENCES = {
    "convaiinnovations/laya": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "convaiinnovations/laya-typed-decisions": "1a793eb568e6718f15941d08f85432581df534e3",
}


def main():
    output = Path("runs/foundation-v11-laya/sources.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = json.loads(output.read_text()) if output.exists() else []
    for repo, revision in REFERENCES.items():
        existing = next((source for source in sources if source["repo"] == repo), None)
        if existing:
            if existing["revision"] != revision:
                raise ValueError("reference revision differs from the frozen experiment")
            for name, expected in existing["sha256"].items():
                if (
                    hashlib.sha256((Path(existing["path"]) / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise ValueError("a pinned reference file changed")
            continue
        folder = Path(
            snapshot_download(
                repo,
                revision=revision,
                token=False,
                cache_dir=".cache/laya-models",
                allow_patterns=[
                    "rl_agent_config.json",
                    "model.safetensors",
                    "tokenizer/*",
                    "encoder/*",
                    "README.md",
                ],
            )
        )
        hashes = {
            str(p.relative_to(folder)).replace("\\", "/"): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in folder.rglob("*")
            if p.is_file()
        }
        sources.append(
            {
                "repo": repo,
                "revision": revision,
                "path": str(folder),
                "sha256": hashes,
                "sdk": "laya==0.3.20",
            }
        )
        output.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"repo": repo, "revision": revision, "files": len(hashes)}), flush=True)


if __name__ == "__main__":
    main()
