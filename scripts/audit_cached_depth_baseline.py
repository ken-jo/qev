"""Recompute the reused Foundation and parent development evidence without GPU inference."""

import json
from pathlib import Path

import torch
from curriculum_release_checks import digest, read_plain, recompute_development, require
from safetensors.torch import load_file
from train_foundation_head import write

from veyra.data import TrainingRecord
from veyra.workflow_learning import annotate


def main():
    output = Path("runs/workflow-v12-depth-plan/cached-development-audit.json")
    if output.exists():
        raise FileExistsError("cached development audit is immutable")
    config = read_plain("configs/workflow-depth-study-v12.json")
    root = Path(config["features"])
    cache = read_plain(root / "manifest.json")
    for filename, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
    ):
        require(digest(root / filename) == cache[key], "cached feature data changed")
    records = []
    for line in Path(config["records"]).open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] in {"train", "dev"}:
            records.append(TrainingRecord.model_validate(raw))
    rows = read_plain(root / "records.json")
    annotate(rows, records)
    tensors = load_file(root / "features.safetensors")
    indices = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    prior_path = Path(config["prior_selection"])
    require(digest(prior_path) == config["prior_selection_sha256"], "prior selection changed")
    prior = read_plain(prior_path)
    baseline = read_plain(config["baseline_development"])
    require(
        digest(config["baseline_development"]) == config["baseline_development_sha256"],
        "baseline changed",
    )
    torch.set_num_threads(4)
    results = []
    for report in (baseline, prior["selected"]):
        metrics, risk = recompute_development(report, tensors, rows, indices)
        results.append(
            {
                "checkpoint": report["checkpoint"],
                "weights_sha256": report["weights_sha256"],
                "development_logits_sha256": report["development_logits_sha256"],
                "novel_workflow_accuracy": metrics["by_domain"]["workflow_new"]["accuracy"],
                "risk": risk,
                "reported_metrics_reproduced": True,
            }
        )
    result = {
        "passed": True,
        "source_sha256": digest(Path(__file__)),
        "auditor_sha256": digest("scripts/curriculum_release_checks.py"),
        "feature_cache": cache,
        "results": results,
        "model_inference_used": False,
        "calibration_or_final_used": False,
        "release_allowed": False,
    }
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
