"""Run real GPU text and photograph inference from the wheel bundled in an SDK package.

Run with `python -I` so only the bundled wheel is importable. The photograph request is the
package's own `examples/photograph/request.json`. With `--previous-record`, every answer is
compared with the earlier package's verification record.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from importlib import metadata
from pathlib import Path

PHOTO_SHA256 = "00e0bbcc0a75007fb79cb8e2e96a48c3d32941ee9ae8918590f846261d5aaac1"
FORBIDDEN = re.compile(r"veyra|[A-Za-z]:\\|/home/|/mnt/|/Users/|/tmp/", re.IGNORECASE)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(new, old):
    """Per-request comparison of answers with an earlier record."""
    report = {}
    for name, entry in new.items():
        before = old["results"][name]["response"]
        after = entry["response"]
        largest = 0.0
        identical = True
        for question, answer in after["answers"].items():
            prior = before["answers"][question]
            if set(prior["probabilities"]) != set(answer["probabilities"]):
                raise ValueError("Candidate set changed: " + question)
            for key, value in answer["probabilities"].items():
                largest = max(largest, abs(value - prior["probabilities"][key]))
            if answer != prior:
                identical = False
        report[name] = {
            "answers_identical": identical,
            "max_abs_probability_difference": largest,
            "selected_answers_identical": all(
                answer.get(field) == before["answers"][question].get(field)
                for question, answer in after["answers"].items()
                for field in ("choice", "abstained")
            ),
            "input_tokens_equal": after["usage"]["input_tokens"] == before["usage"]["input_tokens"],
            "usage_fields_added": sorted(set(after["usage"]) - set(before["usage"])),
            "model_id_changed": before["model"] != after["model"],
            "model": after["model"],
            "response_fields_identical_except_model_answers_usage": {
                k: v for k, v in after.items() if k not in {"model", "answers", "usage"}
            }
            == {k: v for k, v in before.items() if k not in {"model", "answers", "usage"}},
        }
    return report


def locked_versions(lock_text, names):
    versions = {}
    for block in lock_text.split("[[package]]"):
        name = re.search(r'^name = "([^"]+)"', block.strip(), re.MULTILINE)
        version = re.search(r'^version = "([^"]+)"', block, re.MULTILINE)
        if name and version and name.group(1) in names:
            versions.setdefault(name.group(1), set()).add(version.group(1))
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--previous-record", type=Path)
    parser.add_argument("--built-wheel", type=Path, help="the wheel that was packaged")
    parser.add_argument(
        "--control",
        type=Path,
        help="answers from another packaged wheel run in this same environment",
    )
    parser.add_argument(
        "--require-example-match",
        action="store_true",
        help="fail unless the packaged photograph response equals this run",
    )
    args = parser.parse_args()
    if not sys.flags.isolated:
        raise RuntimeError("run Python with -I to isolate the package from the source directory")
    if args.output.exists():
        raise FileExistsError("use a new verification report")
    folder = args.package.resolve()
    checksums = json.loads((folder / "checksums.json").read_text("utf-8"))
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if files != set(checksums) | {"checksums.json"}:
        raise ValueError("package has unlisted or missing files")
    for name, expected in checksums.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or digest(path) != expected:
            raise ValueError("package checksum mismatch: " + name)
    publication = json.loads((folder / "publication.json").read_text("utf-8"))
    wheel = (folder / publication["runtime_wheel_file"]).resolve()
    if not wheel.is_relative_to(folder) or not wheel.is_file():
        raise ValueError("invalid declared runtime wheel")
    wheel_sha = digest(wheel)
    if (
        wheel_sha != checksums[publication["runtime_wheel_file"]]
        or wheel_sha != publication["runtime_wheel_sha256"]
    ):
        raise ValueError("packaged wheel differs from its declared hash")
    if args.built_wheel and digest(args.built_wheel) != wheel_sha:
        raise ValueError("packaged wheel differs from the built wheel")
    sys.path.insert(0, str(wheel))
    import torch

    import qev
    from qev_runtime.option_model import OptionModel
    from qev_runtime.schema import DecisionRequest

    if not str(qev.__file__).startswith(str(wheel)):
        raise RuntimeError("QEV facade was imported outside the bundled wheel")
    if qev.QEV is not OptionModel:
        raise RuntimeError("Public facade changed the evaluated model class")
    photo_root = folder / "examples/photograph"
    if digest(photo_root / "item.jpg") != PHOTO_SHA256:
        raise ValueError("photograph fixture changed")
    photo = DecisionRequest.from_json((photo_root / "request.json").read_text("utf-8"))
    if list(photo.questions) != ["score", "noul", "choice"]:
        raise ValueError("photograph fixture must keep the score, noul, choice order")
    requests = {
        "text": (
            DecisionRequest.from_json((folder / "sample_request.json").read_text("utf-8")),
            folder,
        ),
        "photograph_three_questions": (photo, photo_root),
    }
    torch.set_num_threads(4)
    model = qev.QEV.load(folder, cache_dir=args.cache_dir, local_files_only=True, merge=True)
    forwards = []
    hook = model.encoder.model.register_forward_hook(lambda *_: forwards.append(1))
    results = {}
    try:
        for name, (request, image_root) in requests.items():
            before = len(forwards)
            answer = model.predict(request, image_root)
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
    modules = {}
    for name, module in sorted(sys.modules.items()):
        if name == "qev" or name.startswith(("qev.", "qev_runtime")):
            path = str(getattr(module, "__file__", ""))
            if not path.startswith(str(wheel) + "/"):
                raise RuntimeError("source-directory modules leaked into wheel verification")
            modules[name] = path[len(str(wheel)) + 1 :]
    legacy = sorted(n for n in sys.modules if n.split(".")[0].lower().startswith("veyra"))
    if legacy:
        raise RuntimeError("legacy runtime modules were imported: " + ", ".join(legacy))
    example = json.loads((photo_root / "response.json").read_text("utf-8"))
    example_matches = example == results["photograph_three_questions"]["response"]
    if args.require_example_match and not example_matches:
        raise ValueError("packaged photograph response differs from this run")
    packages = (
        "torch",
        "transformers",
        "pillow",
        "tokenizers",
        "safetensors",
        "accelerate",
        "numpy",
    )
    installed = {name: metadata.version(name) for name in packages}
    locked = locked_versions((folder / "reproducibility/uv.lock").read_text("utf-8"), set(packages))
    result = {
        "passed": True,
        "published_to_huggingface": False,
        "source_sha256": digest(Path(__file__)),
        "weights_sha256": digest(folder / "head.safetensors"),
        "manifest_sha256": digest(folder / "manifest.json"),
        "checksums_sha256": digest(folder / "checksums.json"),
        "verified_files": len(checksums),
        "wheel_file": publication["runtime_wheel_file"],
        "wheel_sha256": wheel_sha,
        "built_wheel_matches_packaged_wheel": bool(args.built_wheel),
        "qev_version": qev.__version__,
        "public_facade_preserves_model_class": True,
        "python_isolated": True,
        "modules_from_bundled_wheel": modules,
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "dependency_versions": installed,
        "dependencies_match_packaged_uv_lock": all(
            installed[name] in locked.get(name, set()) for name in packages
        ),
        "base_model": "Pinned locally cached Qwen base; no fresh base-weight download",
        "gpu": torch.cuda.get_device_name(),
        "photo_request": "examples/photograph/request.json",
        "photo_sha256": PHOTO_SHA256,
        "packaged_example_response_matches": example_matches,
        "results": results,
    }
    if args.control:
        control = json.loads(args.control.read_text("utf-8"))
        control_wheel = (folder / control["wheel_file"]).resolve()
        if not control_wheel.is_relative_to(folder) or control_wheel == wheel:
            raise ValueError("control must use another wheel of this package")
        result["same_environment_control"] = {
            "wheel_file": control["wheel_file"],
            "wheel_sha256": digest(control_wheel),
            "answers_identical": {
                name: control["answers"][name] == entry["response"]["answers"]
                for name, entry in results.items()
            },
        }
    if args.previous_record:
        previous = json.loads(args.previous_record.read_text("utf-8"))
        result["comparison_with_previous_record"] = {
            "previous_qev_version": previous["qev_version"],
            "previous_wheel_sha256": previous["wheel_sha256"],
            "weights_and_manifest_identical": previous["weights_sha256"] == result["weights_sha256"]
            and previous["manifest_sha256"] == result["manifest_sha256"],
            "requests": compare(results, previous),
        }
    text = json.dumps(result, indent=2) + "\n"
    if FORBIDDEN.search(text):
        raise ValueError("report contains a legacy name or a local absolute path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"passed": True, "files": len(checksums), "requests": len(results)}))


if __name__ == "__main__":
    main()
