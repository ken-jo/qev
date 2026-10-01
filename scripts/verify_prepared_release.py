"""Exercise packaged wheel code and checkpoint with real GPU text/image inference."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        raise RuntimeError("run Python with -I to isolate the package from the source directory")
    if args.output.exists():
        raise FileExistsError("use a new verification report")
    folder = args.package.resolve()
    checksums = json.loads((folder / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or digest(path) != expected:
            raise ValueError("package checksum mismatch: " + name)
    wheels = list((folder / "runtime").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one runtime wheel")
    wheel = wheels[0]
    sys.path.insert(0, str(wheel))
    import torch

    import veyra
    import vision_qev
    from veyra.option_model import OptionModel
    from veyra.schema import DecisionRequest

    if not str(veyra.__file__).startswith(str(wheel)):
        raise RuntimeError("Veyra was imported from outside the bundled wheel")
    if not str(vision_qev.__file__).startswith(str(wheel)):
        raise RuntimeError("Vision QEV facade was imported outside the bundled wheel")
    if vision_qev.VisionQEV is not OptionModel:
        raise RuntimeError("Public facade changed the evaluated model class")
    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    photographs = sorted(
        (row for row in records if row["split"] == "test" and "domain:image_waste" in row["tags"]),
        key=lambda row: row["id"],
    )
    first = photographs[0]
    views = [r for r in photographs if r["group_id"] == first["group_id"]]
    # Exact duplicate photographs can share a group; take one question for each typed view.
    questions = {}
    for row in views:
        for question in row["request"]["questions"].values():
            questions.setdefault(question["type"], question)
    if set(questions) != {"choice", "score", "noul"}:
        raise ValueError("photograph fixture must exercise all three types")
    requests = {
        "text": DecisionRequest.from_json((folder / "sample_request.json").read_text("utf-8")),
        "photograph_three_questions": DecisionRequest.model_validate(
            {"state": first["request"]["state"], "questions": questions}
        ),
    }
    torch.set_num_threads(4)
    model = vision_qev.VisionQEV.load(folder, local_files_only=True, merge=True)
    forwards = []
    hook = model.encoder.model.register_forward_hook(lambda *_: forwards.append(1))
    results = {}
    try:
        for name, request in requests.items():
            before = len(forwards)
            answer = model.predict(request, args.records.parent.resolve())
            count = len(forwards) - before
            if count != 1 or answer["usage"]["output_tokens"] != 0:
                raise ValueError("single-forward, zero-generated-token contract failed")
            for question in answer["answers"].values():
                values = list(question["probabilities"].values())
                if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
                    raise ValueError("invalid probabilities")
                if abs(sum(values) - 1) > 1e-5:
                    raise ValueError("probabilities do not sum to one")
            results[name] = {"backbone_forwards": count, "response": answer}
    finally:
        hook.remove()
    modules = {
        name: str(getattr(module, "__file__", ""))
        for name, module in sys.modules.items()
        if name == "veyra" or name.startswith("veyra.")
    }
    if not all(path.startswith(str(wheel)) for path in modules.values()):
        raise RuntimeError("source-directory modules leaked into wheel verification")
    result = {
        "passed": True,
        "published_to_huggingface": False,
        "source_sha256": digest(Path(__file__)),
        "weights_sha256": digest(folder / "head.safetensors"),
        "manifest_sha256": digest(folder / "manifest.json"),
        "checksums_sha256": digest(folder / "checksums.json"),
        "verified_files": len(checksums),
        "wheel_sha256": digest(wheel),
        "veyra_version": veyra.__version__,
        "vision_qev_version": vision_qev.__version__,
        "public_facade_preserves_model_class": True,
        "python_isolated": True,
        "modules_from_bundled_wheel": modules,
        "dependencies": "Existing pinned Python 3.12 environment; not a clean dependency install",
        "base_model": "Pinned locally cached Qwen base; no fresh base-weight download",
        "gpu": torch.cuda.get_device_name(),
        "photo_record_id": first["id"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": True, "files": len(checksums), "requests": len(results)}))


if __name__ == "__main__":
    main()
