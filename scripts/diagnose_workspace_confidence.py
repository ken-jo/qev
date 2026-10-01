"""Compare fixed confidence orderings on saved development outputs without fitting a policy."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_foundation_head import write
from train_workflow_reliability_pilot import rank_report

from veyra.data import TrainingRecord
from veyra.workflow_learning import annotate


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    selection_path = Path("runs/workflow-v12-workspace-selection/selection.json")
    output = selection_path.parent / "confidence-ordering-diagnostic.json"
    if output.exists():
        raise FileExistsError("confidence-ordering diagnosis is immutable")
    config_path = Path("configs/workflow-workspace-study-v12.json")
    config, selection = read(config_path), read(selection_path)
    if selection["study_protocol_sha256"] != digest(config_path):
        raise ValueError("selection does not match the declared continuation study")
    root = Path(config["features"])
    cache, spec = read(root / "manifest.json"), read(root / "specification.json")
    if spec["calibration_or_final_encoded"] is not False:
        raise ValueError("only original train/development representations may be used")
    if digest(config["records"]) != spec["records_sha256"]:
        raise ValueError("original input records changed")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(root / name) != cache[key]:
            raise ValueError("original feature cache changed")
    records = []
    for line in Path(config["records"]).open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] in {"train", "dev"}:
            records.append(TrainingRecord.model_validate(raw))
    tensors, rows = load_file(root / "features.safetensors"), read(root / "records.json")
    annotate(rows, records)
    indices = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    valid, types = tensors["valid"][indices], tensors["types"][indices]
    candidates = [read(selection["baseline_merged_dev"]), *selection["candidates"]]
    if len(candidates) != 6:
        raise ValueError("compare the baseline and all five declared candidates")
    torch.set_num_threads(4)
    results, bindings = [], {}
    for candidate in candidates:
        checkpoint = Path(candidate["checkpoint"])
        path = Path(candidate["development_logits"])
        for file, expected in (
            (path, candidate["development_logits_sha256"]),
            (checkpoint / "manifest.json", candidate["manifest_sha256"]),
            (checkpoint / "head.safetensors", candidate["weights_sha256"]),
        ):
            if digest(file) != expected:
                raise ValueError("measured development candidate changed")
            bindings[str(file)] = expected
        cached = load_file(path)
        if not torch.equal(cached["cache_indices"], indices):
            raise ValueError("development record ordering changed")
        logits = cached["logits"]
        if logits.shape != valid.shape or not torch.isfinite(logits).all():
            raise ValueError("invalid stored development logits")
        temperatures = read(checkpoint / "manifest.json")["calibration"]["temperatures"]
        scaled = logits / torch.tensor(temperatures)[types, None]
        probabilities = scaled.masked_fill(~valid, -1e9).softmax(-1)
        top = probabilities.topk(2, dim=-1).values
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        confidence = {
            "maximum_probability": top[:, 0],
            "probability_margin": top[:, 0] - top[:, 1],
            "top_two_conditional_probability": top[:, 0] / top.sum(-1).clamp_min(1e-12),
            "one_minus_normalized_entropy": (1 - entropy / valid.sum(-1).float().log()).clamp(0, 1),
        }
        measured = {}
        for name, score in confidence.items():
            result = rank_report(score, logits, tensors, indices, rows)
            for values in result["by_type"].values():
                # Ranking scores other than maximum probability are not reported probabilities.
                values.pop("correctness_brier")
            measured[name] = result
        for kind, expected in candidate["development_risk_diagnostic"].items():
            observed = measured["maximum_probability"]["by_type"][kind]
            if (
                observed["maximum_threshold_coverage_at_15pct_error"]
                != expected["maximum_coverage_at_15pct_error"]
            ):
                raise ValueError("original maximum-probability coverage did not reproduce")
        results.append({"checkpoint": str(checkpoint), "orderings": measured})
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": digest(selection_path),
        "source_sha256": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("scripts/train_workflow_reliability_pilot.py"),
                Path("src/veyra/workflow_learning.py"),
            )
        },
        "bindings": bindings,
        "feature_cache": cache,
        "results": results,
        "used_for_candidate_selection": False,
        "policy_fitted_or_runtime_changed": False,
        "calibration_or_final_used": False,
        "release_allowed": False,
        "limitations": (
            "Exploratory development diagnostics of four fixed orderings. Model predictions, "
            "probability distributions, NLL and distribution Brier stay unchanged. The other "
            "scores are ranking statistics, not calibrated probabilities of correctness. "
            "Fixed-count 60% risk can split ties; maximum threshold coverage keeps ties together. "
            "This analysis does not adopt any alternative policy or add a release candidate. "
            "Any later alternative requires a separate declaration, fresh calibration and "
            "the same full release gates before final evaluation."
        ),
    }
    write(output, report)
    print(json.dumps({"output": str(output), "checkpoints": len(results)}), flush=True)


if __name__ == "__main__":
    main()
