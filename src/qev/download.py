"""Download the immutable QEV decision checkpoint and its upstream backbone."""

import hashlib
from pathlib import Path

CHECKPOINT_ID = "ken-jo/qev"
# The repository rename preserves this original checkpoint revision.
CHECKPOINT_REVISION = "0d2d13ffb3c392071ea00b6bdb903c4b84dff48d"
CHECKPOINT_HASHES = {
    "head.safetensors": "84b57aeeb987f73416ac0f796150957d1c5beb1ed373733053e46438f77a8fee",
    "manifest.json": "d83f9910196c6e801658ca3816c3d2bd4a849ba3da6ff059a0edf7c6028e898d",
}


def download_checkpoint(output: Path, cache_dir: str) -> Path:
    """Copy verified checkpoint files into an explicit local directory."""
    from huggingface_hub import hf_hub_download

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in CHECKPOINT_HASHES.items():
        destination = output / name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Refusing to replace a different checkpoint file: {destination}")
            continue
        source = Path(
            hf_hub_download(
                CHECKPOINT_ID,
                name,
                revision=CHECKPOINT_REVISION,
                cache_dir=cache_dir,
                token=False,
            )
        )
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Checkpoint integrity check failed: {name}")
        with destination.open("xb") as stream:
            stream.write(raw)
    return output


def download_backbone(cache_dir: str) -> str:
    """Download the exact upstream weights and preprocessing files."""
    from huggingface_hub import snapshot_download

    from veyra.constants import MODEL_ID, MODEL_REVISION

    return snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
        token=False,
    )
