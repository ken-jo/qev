"""Compare exploratory heads on development risk and audit primitive-factor generalization."""

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_foundation_head import write

from veyra.constants import QUESTION_TYPES
from veyra.transfer_learning import TransferReadout


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@torch.inference_mode()
def main():
    output = Path("runs/workflow-v12-pilot-diagnostics.json")
    if output.exists():
        raise FileExistsError("pilot diagnosis is immutable")
    features = Path("data/features-workflow-v12")
    tensors = load_file(features / "features.safetensors")
    rows = read(features / "records.json")
    dev = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "dev"])
    novel = torch.tensor([rows[i]["domain"] == "workflow_new" for i in dev.tolist()])
    candidates = [
        read("runs/workflow-v12-head/selection.json")["selected"],
        read("runs/workflow-v12-consistency-pilot/comparison.json")["selected"],
        read("runs/workflow-v12-facts-pilot/comparison.json")["selected"],
    ]
    reports = []
    torch.set_num_threads(4)
    for candidate in candidates:
        path = Path(candidate["checkpoint"])
        manifest = read(path / "manifest.json")
        model = TransferReadout(manifest, load_file(path / "head.safetensors")).eval()
        values = model(tensors["states"][dev], tensors["condition_logits"][dev])
        types, valid, gold = tensors["types"][dev], tensors["valid"][dev], tensors["gold"][dev]
        scale = torch.tensor(manifest["calibration"]["temperatures"])[types, None]
        probabilities = (values / scale).masked_fill(~valid, -1e9).softmax(-1)
        confidence, prediction = probabilities.max(-1)
        expected_error = 1 - tensors["targets"][dev].gather(1, prediction[:, None]).squeeze(1)
        expected_error = torch.where(gold >= 0, (prediction != gold).float(), expected_error)
        by_type = {}
        for type_id, kind in enumerate(QUESTION_TYPES):
            indices = torch.where(types == type_id)[0]
            order = indices[torch.argsort(confidence[indices], descending=True, stable=True)]
            count = len(order)
            prefix_counts = torch.arange(1, count + 1)
            risks = expected_error[order].cumsum(0) / prefix_counts
            boundaries = torch.ones(count, dtype=torch.bool)
            boundaries[:-1] = confidence[order[:-1]] != confidence[order[1:]]
            eligible = torch.where((risks <= 0.15) & (prefix_counts >= 20) & boundaries)[0]
            accepted = int(eligible[-1]) + 1 if len(eligible) else 0
            novel_type = (types == type_id) & novel
            by_type[kind] = {
                "questions": count,
                "new_workflow_accuracy": float(
                    (prediction[novel_type] == gold[novel_type]).float().mean()
                ),
                "risk_at_60pct_coverage": float(risks[max(0, math.floor(count * 0.6) - 1)]),
                "maximum_coverage_at_15pct_error": accepted / count,
            }
        result = {
            "method": candidate["method"],
            "checkpoint": str(path),
            "weights_sha256": digest(path / "head.safetensors"),
            "by_type": by_type,
        }
        if candidate["method"].startswith("rloo-facts-"):
            auxiliary = load_file("runs/workflow-v12-facts-pilot/auxiliary_targets.safetensors")
            decoder = torch.nn.Linear(manifest["binding_head"]["rank"], 4)
            decoder.load_state_dict(load_file(path.parent / "auxiliary-training-only.safetensors"))
            indices = {
                split: torch.tensor(
                    [
                        i
                        for i, r in enumerate(rows)
                        if r["split"] == split and auxiliary["eligible"][i]
                    ]
                )
                for split in ("train", "dev")
            }
            train_mean = auxiliary["targets"][indices["train"]].mean(0)
            fact_report = {}
            for split, ix in indices.items():
                hidden = torch.nn.functional.gelu(
                    model.binding_head.project(model.binding_head.normalize(tensors["states"][ix]))
                )
                prediction = decoder(hidden).sigmoid()
                target = auxiliary["targets"][ix]
                hard = (target == 0) | (target == 1)
                fact_report[split] = {
                    "questions": len(ix),
                    "condition_probability_mse": float((prediction - target).square().mean()),
                    "constant_half_mse": float((target - 0.5).square().mean()),
                    "training_marginal_mse": float((target - train_mean).square().mean()),
                    "observed_condition_accuracy": float(
                        ((prediction[hard] >= 0.5) == (target[hard] == 1)).float().mean()
                    ),
                    "training_marginal_observed_accuracy": float(
                        ((train_mean.expand_as(target)[hard] >= 0.5) == (target[hard] == 1))
                        .float()
                        .mean()
                    ),
                }
            result["auxiliary_condition_diagnosis"] = fact_report
        reports.append(result)
    result = {
        "scope": "Unmerged cached features; development diagnosis, not final release evidence",
        "source_sha256": digest(Path(__file__)),
        "calibration_or_final_used": False,
        "deployed_policy_changed": False,
        "candidates": reports,
    }
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
