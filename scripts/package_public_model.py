"""Repackage the verified checkpoint with the renamed runtime and English model card."""

import argparse
import hashlib
import json
import re
import shutil
import tomllib
import zipfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = json.loads((args.prepared / "checksums.json").read_text("utf-8"))
    for name, sha in expected.items():
        if digest(args.prepared / name) != sha:
            raise ValueError("Original prepared package changed: " + name)
    wheel = root / f"dist/runtime/qev-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        modules = sorted((root / "src/veyra").glob("*.py"))
        for module in modules:
            if archive.read("veyra/" + module.name) != module.read_bytes():
                raise ValueError("Renamed wheel runtime differs from frozen source")
    skip = {
        "README.md",
        "ROADMAP.ko.md",
        "ROADMAP.md",
        "checksums.json",
        "NOTICE",
        "load_veyra.py",
        "load_vision_qev.py",
        "load_qwen3_5_classification.py",
        "load_qev.py",
        "upload_to_hub.py",
    }
    for path in sorted(args.prepared.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(args.prepared)
        if relative.as_posix() in skip or relative.parts[0] in {"runtime", "reproducibility"}:
            continue
        if path.suffix == ".md" and re.search("[\uac00-\ud7a3]", path.read_text("utf-8")):
            raise ValueError("Private-language document in model payload")
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ("LICENSE", "NOTICE", "ROADMAP.md"):
        shutil.copy2(root / name, args.output / name)
    shutil.copy2(root / "MODEL_CARD.md", args.output / "README.md")
    for directory in ("runtime", "reproducibility"):
        (args.output / directory).mkdir(exist_ok=True)
    shutil.copy2(wheel, args.output / "runtime" / wheel.name)
    previous_wheel = root / "dist/runtime/qev-0.1.1-py3-none-any.whl"
    if previous_wheel.exists() and previous_wheel != wheel:
        shutil.copy2(previous_wheel, args.output / "runtime" / previous_wheel.name)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(root / name, args.output / "reproducibility" / name)
    shutil.copytree(root / "reports/release-comparison", args.output / "release-comparison")
    (args.output / "docs").mkdir(exist_ok=True)
    for name in (
        "LAYA_COMPARISON.md",
        "MODEL_SIZE.md",
        "COMPACT_MODEL.md",
        "PLAYGROUND.md",
        "API.md",
    ):
        shutil.copy2(root / "docs" / name, args.output / "docs" / name)
    shutil.copy2(root / "examples/request.json", args.output / "sample_request.json")
    photo = args.output / "examples/photograph"
    photo.mkdir(parents=True, exist_ok=True)
    for name in (
        "item.jpg",
        "request.json",
        "response.json",
        "provenance.json",
        "README.md",
        "LICENSE.txt",
    ):
        shutil.copy2(root / "examples/photograph" / name, photo / name)
    (args.output / "load_qev.py").write_text(
        '"""Run the packaged example with the latest bundled qev wheel."""\n'
        "import json\nfrom pathlib import Path\nimport torch\n"
        "from qev import load, DecisionRequest\n\n"
        "root = Path(__file__).resolve().parent\n"
        "torch.set_num_threads(4)\n"
        "model = load(root)\n"
        'request = DecisionRequest.from_json((root / "sample_request.json").read_text("utf-8"))\n'
        "print(json.dumps(model.predict(request, root), indent=2))\n",
        encoding="utf-8",
    )
    publication = {
        "project": "qev",
        "version": version,
        "model_version": "0.1.1",
        "runtime_wheel_file": "runtime/" + wheel.name,
        "source_repository": "https://github.com/ken-jo/qev",
        "backbone": "Qwen/Qwen3.5-2B",
        "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "weights_sha256": digest(args.output / "head.safetensors"),
        "manifest_sha256": digest(args.output / "manifest.json"),
        "runtime_wheel_sha256": digest(wheel),
        "preserved_runtime_modules": len(modules),
        "original_prepared_checksums_sha256": digest(args.prepared / "checksums.json"),
        "original_acceptance_preserved": True,
        "weights_calibration_and_runtime_inference_unchanged": True,
        "separate_publication_verification": "release/verification.json in the source repository",
    }
    (args.output / "publication.json").write_text(
        json.dumps(publication, indent=2) + "\n", encoding="utf-8"
    )
    checksums = {
        p.relative_to(args.output).as_posix(): digest(p)
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    }
    (args.output / "checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(publication))


if __name__ == "__main__":
    main()
