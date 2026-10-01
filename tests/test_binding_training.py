import random
import runpy
from pathlib import Path

import torch

from veyra.binding_head import ConditionedReadout

scripts = Path(__file__).resolve().parents[1] / "scripts"
evaluate = runpy.run_path(str(scripts / "train_binding_head.py"))["evaluate"]
controlled_training = runpy.run_path(str(scripts / "cache_binding_features.py"))[
    "controlled_training"
]
pair = runpy.run_path(str(Path(__file__).with_name("test_workspace_diagnostic.py")))["pair"]


def test_training_feature_requests_exclude_all_held_out_records():
    train = pair("choice", 1, "train")
    heldout = [
        record for split in ("dev", "calibration", "test") for record in pair("noul", 1, split)
    ]
    transformed = controlled_training(train + heldout, random.Random(43))
    assert len(transformed) == 2
    assert all(record.split == "train" for record in transformed)
    assert all(
        "oracle_condition_branch" not in record.request.model_dump_json() for record in transformed
    )


def test_cached_development_metrics_mask_invalid_option_letters():
    head = ConditionedReadout(rank=4, input_size=8)
    valid = torch.zeros(2, 16, dtype=torch.bool)
    valid[:, [2, 5]] = True
    base = torch.zeros(2, 16)
    base[:, 0] = 1000  # An unavailable option must never win.
    base[:, 5] = 3
    targets = torch.zeros(2, 16)
    targets[:, 5] = 1
    tensors = {
        "states": torch.randn(2, 8),
        "condition_logits": torch.randn(2, 3),
        "base_logits": base,
        "valid": valid,
        "targets": targets,
    }
    rows = [
        {"family": "text_ranges", "type": "choice"},
        {"family": "image_count_rule", "type": "choice"},
    ]
    result = evaluate(head, tensors, rows, torch.tensor([0, 1]))
    assert result["text_accuracy"] == result["image_accuracy"] == 1
    assert 0 < result["nll"] < 0.1
