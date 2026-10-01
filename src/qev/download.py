"""Download the immutable QEV decision checkpoint and its upstream backbone."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

CHECKPOINT_ID = "ken-jo/qev"
# The repository rename preserves this original checkpoint revision.
CHECKPOINT_REVISION = "0d2d13ffb3c392071ea00b6bdb903c4b84dff48d"
CHECKPOINT_HASHES = {
    "head.safetensors": "84b57aeeb987f73416ac0f796150957d1c5beb1ed373733053e46438f77a8fee",
    "manifest.json": "d83f9910196c6e801658ca3816c3d2bd4a849ba3da6ff059a0edf7c6028e898d",
}


def download_checkpoint(output: Path, cache_dir: str, *, local_files_only=False) -> Path:
    """Copy verified checkpoint files into an explicit local directory."""
    from filelock import FileLock
    from huggingface_hub import hf_hub_download

    output = output.resolve()
    existing = []
    for name, expected in CHECKPOINT_HASHES.items():
        target = output / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Refusing to replace a different checkpoint file: {target}")
            existing.append(name)
    if len(existing) == len(CHECKPOINT_HASHES):
        return output
    output.mkdir(parents=True, exist_ok=True)
    locks = Path(cache_dir) / ".qev-locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(os.path.normcase(str(output)).encode()).hexdigest() + ".lock"
    with FileLock(str(locks / lock_name)):
        for name, expected in CHECKPOINT_HASHES.items():
            destination = output / name
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                    raise ValueError(
                        f"Refusing to replace a different checkpoint file: {destination}"
                    )
                continue
            source = Path(
                hf_hub_download(
                    CHECKPOINT_ID,
                    name,
                    revision=CHECKPOINT_REVISION,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    token=False,
                )
            )
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Checkpoint integrity check failed: {name}")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=output, suffix=".partial", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return output


def download_backbone(cache_dir: str, *, local_files_only=False) -> str:
    """Download the exact upstream weights and preprocessing files."""
    from huggingface_hub import snapshot_download

    from veyra.constants import MODEL_ID, MODEL_REVISION

    folder = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
            token=False,
        )
    )
    required = {
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "video_preprocessor_config.json",
        "model.safetensors.index.json",
    }
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        required.update(json.loads(index.read_text("utf-8"))["weight_map"].values())
    missing = []
    for name in required:
        # Hub snapshots may symlink weights into the adjacent blob cache.
        path = folder / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not path.is_file()
            or path.stat().st_size == 0
        ):
            missing.append(name)
    if missing:
        from huggingface_hub.errors import LocalEntryNotFoundError

        raise LocalEntryNotFoundError("Incomplete Qwen snapshot: " + ", ".join(sorted(missing)))
    return str(folder)
