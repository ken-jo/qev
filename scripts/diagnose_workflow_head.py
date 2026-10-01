"""Inspect development-only error ranking without fitting a deployed abstention policy."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_foundation_head import write

from veyra.constants import QUESTION_TYPES
from veyra.transfer_learning import TransferReadout


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    output = Path("runs/workflow-v12-head/development-risk-diagnostic.json")
    if output.exists():
        raise FileExistsError("development diagnosis is immutable")
    selected = json.loads(Path("runs/workflow-v12-head/selection.json").read_text())["selected"]
    checkpoint = Path(selected["checkpoint"])
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    tensors = load_file("data/features-workflow-v12/features.safetensors")
    rows = json.loads(Path("data/features-workflow-v12/records.json").read_text())
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    types, valid = tensors["types"][dev], tensors["valid"][dev]
    model = TransferReadout(manifest, load_file(checkpoint / "head.safetensors")).eval()
    torch.set_num_threads(4)
    values = model(tensors["states"][dev], tensors["condition_logits"][dev])
    scale = torch.tensor(manifest["calibration"]["temperatures"])[types, None]
    probabilities = (values / scale).masked_fill(~valid, -1e9).softmax(-1)
    confidence, prediction = probabilities.max(-1)
    gold = tensors["gold"][dev]
    errors = 1 - tensors["targets"][dev].gather(1, prediction[:, None]).squeeze(1)
    errors = torch.where(gold >= 0, (prediction != gold).float(), errors)
    diagnostics = {}
    for type_id, kind in enumerate(QUESTION_TYPES):
        indices = torch.where(types == type_id)[0]
        order = indices[torch.argsort(confidence[indices], descending=True, stable=True)]
        n = len(order)
        accepted_counts = torch.arange(1, n + 1)
        risks = errors[order].cumsum(0) / accepted_counts
        boundaries = torch.ones(n, dtype=torch.bool)
        boundaries[:-1] = confidence[order[:-1]] != confidence[order[1:]]
        eligible = torch.where((risks <= 0.15) & (accepted_counts >= 20) & boundaries)[0]
        last = int(eligible[-1]) if len(eligible) else None
        diagnostics[kind] = {
            "questions": n,
            "maximum_development_coverage_at_15pct_error": (last + 1) / n
            if last is not None
            else 0,
            "risk_at_maximum_coverage": float(risks[last]) if last is not None else None,
            "risk_at_60pct_coverage": float(risks[max(0, int(n * 0.6) - 1)]),
            "risk_at_80pct_coverage": float(risks[max(0, int(n * 0.8) - 1)]),
            "risk_at_full_coverage": float(risks[-1]),
        }
    result = {
        "scope": "Unmerged cached-feature development diagnosis only; not release evidence",
        "weights_sha256": digest(checkpoint / "head.safetensors"),
        "source_sha256": digest(Path(__file__)),
        "calibration_or_final_used": False,
        "deployed_policy_changed": False,
        "by_type": diagnostics,
    }
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
