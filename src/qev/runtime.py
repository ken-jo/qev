"""First-use downloads and device selection around the frozen QEV model."""

import os
from pathlib import Path

from qev.download import CHECKPOINT_REVISION, download_backbone, download_checkpoint


def cache_home():
    explicit = os.environ.get("QEV_HOME")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "qev"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "qev"


def resolve_cache(cache_dir=None):
    return str(Path(cache_dir or os.environ.get("QEV_CACHE_DIR") or cache_home() / "hub"))


def resolve_device(device="auto"):
    import torch

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Choose auto, cpu or cuda.")
    cuda = torch.cuda.is_available()
    if device == "cuda" and not cuda:
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch or use --device cpu."
        )
    return ("cuda" if cuda else "cpu") if device == "auto" else device


def prepare_checkpoint(checkpoint=None, *, cache_dir=None, offline=False, progress=None):
    """Resolve a complete pinned model; contact the Hub only for missing files."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    folder = Path(checkpoint) if checkpoint else cache_home() / "checkpoints" / CHECKPOINT_REVISION
    cache = resolve_cache(cache_dir)
    offline = offline or os.environ.get("QEV_OFFLINE") == "1"
    try:
        download_checkpoint(folder, cache, local_files_only=True)
        download_backbone(cache, local_files_only=True)
    except LocalEntryNotFoundError as error:
        if offline:
            raise RuntimeError(
                "QEV is not fully cached. Run qev download once with network access, "
                "using the same --checkpoint/--cache-dir or QEV_HOME."
            ) from error
        if progress:
            progress("Downloading QEV and its pinned Qwen backbone (about 4.6 GB on first use).")
        download_checkpoint(folder, cache)
        download_backbone(cache)
    return folder.resolve(), cache


def load_model(checkpoint=None, *, device="auto", cache_dir=None, offline=False, progress=None):
    from veyra.option_model import OptionModel

    selected = resolve_device(device)
    folder, cache = prepare_checkpoint(
        checkpoint, cache_dir=cache_dir, offline=offline, progress=progress
    )
    if progress:
        progress(f"Loading QEV on {selected.upper()}...")
    return OptionModel.load(
        folder, device=selected, cache_dir=cache, local_files_only=True, merge=True
    )
