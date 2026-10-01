"""Inspect packaging of actual depth-study sources without creating a model release."""

import hashlib
import json
from pathlib import Path

from prepare_workflow_huggingface import evidence_packaging_plan
from train_foundation_head import write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    config_path = Path("configs/workflow-depth-study-v12.json")
    config = read(config_path)
    root = Path.cwd().resolve()
    output = Path("runs/workflow-v12-depth-plan/hf-evidence-inventory.json")
    if output.exists():
        raise FileExistsError("the evidence inventory audit is immutable")
    protocol = read("runs/workflow-v12-depth-study/full_depth/protocol.json")
    features = Path(config["features"])
    paths = {
        **protocol["source_sha256"],
        str(config_path): digest(config_path),
        config["records"]: config["records_sha256"],
        config["training_plan"]: config["training_plan_sha256"],
        str(features / "manifest.json"): config["feature_cache_manifest_sha256"],
        str(features / "features.safetensors"): read(features / "manifest.json")["features_sha256"],
    }
    plan = evidence_packaging_plan(paths, root)
    expected_external = {config["records"], str(features / "features.safetensors")}
    if set(plan["external_inputs"]) != expected_external:
        raise ValueError("original records and feature cache were not kept outside the bundle")
    if set(plan["bundled"]) | set(plan["external_inputs"]) != set(paths):
        raise ValueError("an evidence source was dropped from the inventory")
    if set(plan["bundled"]) & set(plan["external_inputs"]):
        raise ValueError("an evidence source has conflicting packaging roles")
    write(
        output,
        {
            "passed": True,
            "scope": (
                "Actual available depth-study source/input inventory only. "
                "Release acceptance and model packaging remain pending."
            ),
            "configuration_sha256": digest(config_path),
            "source_sha256": {
                str(Path(__file__)): digest(__file__),
                "scripts/prepare_workflow_huggingface.py": digest(
                    "scripts/prepare_workflow_huggingface.py"
                ),
            },
            "input_count": len(paths),
            "bundled_count": len(plan["bundled"]),
            "external_input_count": len(plan["external_inputs"]),
            "external_input_bytes": sum(Path(name).stat().st_size for name in expected_external),
            "plan": plan,
            "all_local_input_hashes_checked": True,
            "original_dataset_content_written": False,
            "release_package_created": False,
            "model_inference_used": False,
            "release_allowed": False,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "bundled": len(plan["bundled"]),
                "external": len(expected_external),
                "release_package_created": False,
            }
        )
    )


if __name__ == "__main__":
    main()
