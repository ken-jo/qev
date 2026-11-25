"""Copy a verified subdir bundled checkpoint to the recommended root layout, offline."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from qev_runtime.bundled_backbone import _LOCAL_METADATA, PACKAGE_FILES, verify_bundle


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _rename_no_replace(source, destination):
    """Publish atomically, including when another process creates an empty DST."""
    if sys.platform == "win32":
        os.rename(source, destination)  # Windows rename already refuses existing paths.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    else:
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise NotImplementedError("atomic rename without replacement is unavailable")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        # AT_FDCWD and RENAME_NOREPLACE; staging and DST share a filesystem.
        arguments = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def flatten(source, destination):
    """Preserve the source, all backbone hashes and head bytes; never overwrite."""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination already exists")
    resolved = destination.resolve()
    if resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError("source and destination must not overlap")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 3:
        raise ValueError("source must be a format-3 bundled checkpoint")
    block = manifest["backbone"]
    if block.get("layout", "subdir") != "subdir":
        raise ValueError("source must use the subdir layout")
    root = verify_bundle(source, block)
    if digest(source / "head.safetensors") != manifest["weights_sha256"]:
        raise ValueError("checkpoint weight checksum mismatch")
    # A subdir inventory must not enter reserved or ignored root locations.
    if any(name.split("/", 1)[0] in PACKAGE_FILES for name in block["files"]):
        raise ValueError("backbone inventory collides with a reserved package file")
    if any(
        name == metadata or name.startswith(metadata + "/")
        for name in block["files"]
        for metadata in _LOCAL_METADATA
    ):
        raise ValueError("backbone inventory collides with an ignored root metadata path")
    converted = manifest | {"backbone": block | {"layout": "root"}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-staging-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        for name in sorted(block["files"]):
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            # Materialize Hub file symlinks so the new package is self-contained.
            shutil.copyfile(root / name, target)
        for name in sorted(PACKAGE_FILES - {"manifest.json"}):
            if (source / name).is_file():
                shutil.copyfile(source / name, staging / name)
        (staging / "manifest.json").write_text(
            json.dumps(converted, indent=2) + "\n", encoding="utf-8"
        )
        written = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        if written != converted:
            raise ValueError("repackaged manifest mismatch")
        verify_bundle(staging, written["backbone"])
        for name in PACKAGE_FILES - {"manifest.json"}:
            if (source / name).is_file() and digest(source / name) != digest(staging / name):
                raise ValueError(f"repackaged file checksum mismatch: {name}")
        if os.name == "posix":
            # TemporaryDirectory starts at 0700; publish with normal mkdir permissions.
            umask = os.umask(0)
            os.umask(umask)
            staging.chmod(0o777 & ~umask)
        _rename_no_replace(staging, destination)
    return {"backbone_files": len(block["files"]), "hashes_identical": True, "head_identical": True}


def compare_packages(source, destination, count):
    """Compare deterministic text requests with both real models on CPU."""
    import torch

    from qev_runtime.option_model import OptionModel
    from qev_runtime.schema import DecisionRequest

    before = OptionModel.load(source, device="cpu", local_files_only=True)
    after = OptionModel.load(destination, device="cpu", local_files_only=True)
    largest = 0.0
    for index in range(count):
        request = DecisionRequest.model_validate(
            {
                "state": {"text": f"Comparison evidence {index}: alpha or beta?"},
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Which option matches the evidence?",
                        "criteria": {"a": "alpha", "b": "beta"},
                    }
                },
            }
        )
        with torch.inference_mode():
            expected = before.predict(request)["answers"]
            actual = after.predict(request)["answers"]
        for question, answer in expected.items():
            for option, value in answer["probabilities"].items():
                largest = max(largest, abs(value - actual[question]["probabilities"][option]))
        if actual != expected or largest != 0.0:
            raise ValueError(f"repackaged outputs differ on request {index}")
    return {"requests": count, "max_abs_probability_difference": largest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--compare", type=int, default=0, metavar="N", help="compare N CPU requests"
    )
    args = parser.parse_args()
    if args.compare < 0:
        parser.error("--compare must be nonnegative")
    result = flatten(args.source, args.destination)
    if args.compare:
        result["comparison"] = compare_packages(args.source, args.destination, args.compare)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
