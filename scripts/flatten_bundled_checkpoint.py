"""Copy a verified subdir bundled checkpoint to the recommended root layout, offline."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from qev_runtime.bundled_backbone import PACKAGE_FILES, verify_bundle


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    # A subdir inventory must not collide with root package files after flattening.
    if PACKAGE_FILES.intersection(block["files"]):
        raise ValueError("backbone inventory collides with a reserved package file")
    converted = manifest | {"backbone": block | {"layout": "root"}}
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(block["files"]):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Materialize Hub file symlinks so the new package is self-contained.
        shutil.copyfile(root / name, target)
    for name in sorted(PACKAGE_FILES - {"manifest.json"}):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    (destination / "manifest.json").write_text(
        json.dumps(converted, indent=2) + "\n", encoding="utf-8"
    )
    written = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if written != converted:
        raise ValueError("repackaged manifest mismatch")
    verify_bundle(destination, written["backbone"])
    for name in PACKAGE_FILES - {"manifest.json"}:
        if (source / name).is_file() and digest(source / name) != digest(destination / name):
            raise ValueError(f"repackaged file checksum mismatch: {name}")
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
