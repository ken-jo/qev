"""Version the new release gate after independent policy validation succeeds."""

import argparse
import json
from pathlib import Path

import torch
from depth_policy_checks import digest, read, require
from recovery_policy_checks import collect_recovery_evidence
from train_foundation_head import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    require(not args.output.exists(), "runtime version transition already exists")
    require(not (args.evaluation / "final-freeze.json").exists(), "final sources already frozen")
    evidence = collect_recovery_evidence(
        args.checkpoint,
        args.selection,
        args.records,
        args.evaluation / "calibration/evaluation.json",
        args.evaluation / "previous-calibration-policy.json",
    )
    require(all(evidence["checks"].values()), "policy validation is incomplete")
    complete = read("runs/backbone-recovery-memory-v13/training-complete.json")
    require(complete["completed"] is True, "recovery training is incomplete")
    changes = {
        Path("pyproject.toml"): (
            b'name = "veyra"\nversion = "0.3.0a3"',
            b'name = "veyra"\nversion = "0.3.0a4"',
        ),
        Path("uv.lock"): (
            b'name = "veyra"\nversion = "0.3.0a3"',
            b'name = "veyra"\nversion = "0.3.0a4"',
        ),
        Path("src/veyra/__init__.py"): (b'__version__ = "0.3.0a3"', b'__version__ = "0.3.0a4"'),
    }
    changes[Path("src/veyra/workflow_release_gate.py")] = (
        b"EXTENSION_CHECKS = {\n",
        b'EXTENSION_CHECKS = {\n    "workflow_recovery_v13": (\n'
        b'        "recovery_study_provenance", "fresh_data_provenance",\n'
        b'        "independent_policy_validation", "previous_calibration_policy_regression",\n'
        b"    ),\n",
    )
    pending, hashes, sources = {}, {}, {}
    archive_root = Path("reports/workflow-v13/runtime-source-snapshots")

    for path, (before, after) in changes.items():
        content = path.read_bytes()
        require(content.count(before) == 1 and after not in content, "unexpected runtime version")
        pending[path] = content.replace(before, after)
        hashes[str(path)] = {"before_sha256": digest(path)}
    for path in pending:
        if path.parts[:2] == ("src", "veyra"):
            archived = archive_root / ("0.3.0a3-" + path.name + ".txt")
            require(not archived.exists(), "runtime source archive already exists")
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(path.read_bytes())
            sources[path.as_posix()] = {"before_sha256": digest(path), "archive": str(archived)}
    for path, content in pending.items():
        path.write_bytes(content)
        hashes[str(path)]["after_sha256"] = digest(path)
        if path.as_posix() in sources:
            sources[path.as_posix()]["after_sha256"] = digest(path)
    write(
        args.output,
        {
            "completed": True,
            "version": "0.3.0a4",
            "selection_sha256": digest(args.selection),
            "source_sha256": digest(__file__),
            "changed_files": hashes,
            "source_changes": sources,
            "policy_checks": evidence["checks"],
            "model_weights_or_training_environment_changed": False,
            "release_allowed": False,
        },
    )
    print(json.dumps({"runtime_version": "0.3.0a4", "published": False}), flush=True)


if __name__ == "__main__":
    main()
