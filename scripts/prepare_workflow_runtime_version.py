"""Advance only the runtime package version after training and eligible selection finish."""

import argparse
import hashlib
import json
from pathlib import Path

from train_foundation_head import write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("runtime version transition is immutable")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    extension = set(selection.get("release_extensions", {}))
    if extension == {"workflow_curriculum_v12"}:
        study = Path("runs/workflow-v12-curriculum-study")
    elif extension == {"workflow_depth_v12"}:
        study = Path("runs/workflow-v12-depth-study")
    else:
        raise ValueError("unsupported runtime version transition study")
    complete = json.loads((study / "complete.json").read_text(encoding="utf-8"))
    if (
        not complete["completed"]
        or not selection["eligible"]
        or complete["configuration_sha256"] != selection["study_protocol_sha256"]
    ):
        raise ValueError("finish training and eligible selection before changing runtime metadata")
    changes = {
        Path("pyproject.toml"): (
            b'name = "veyra"\nversion = "0.3.0a1"',
            b'name = "veyra"\nversion = "0.3.0a2"',
        ),
        Path("uv.lock"): (
            b'name = "veyra"\nversion = "0.3.0a1"',
            b'name = "veyra"\nversion = "0.3.0a2"',
        ),
        Path("src/veyra/__init__.py"): (b'__version__ = "0.3.0a1"', b'__version__ = "0.3.0a2"'),
    }
    pending, hashes = {}, {}
    for path, (before, after) in changes.items():
        content = path.read_bytes()
        if content.count(before) != 1 or after in content:
            raise ValueError("unexpected runtime version metadata: " + str(path))
        pending[path] = content.replace(before, after)
        hashes[str(path)] = {"before_sha256": digest(path)}
    # This script runs before final freeze and never touches the training environment or weights.
    for path, content in pending.items():
        path.write_bytes(content)
        hashes[str(path)]["after_sha256"] = digest(path)
    write(
        args.output,
        {
            "completed": True,
            "version": "0.3.0a2",
            "selection_sha256": digest(args.selection),
            "source_sha256": digest(Path(__file__)),
            "changed_files": hashes,
            "model_weights_or_training_environment_changed": False,
            "release_allowed": False,
        },
    )
    print(json.dumps({"runtime_version": "0.3.0a2", "published": False}), flush=True)


if __name__ == "__main__":
    main()
