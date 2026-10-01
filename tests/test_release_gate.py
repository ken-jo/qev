import copy

import pytest

from veyra.release_gate import audit_release


def evidence():
    calibration_state = {"temperatures": [1.5, 1.5, 1.5]}
    manifest = {
        "weights_sha256": "weights",
        "calibration": calibration_state,
        "training": {
            "completed": True,
            "optimizer_steps": 100,
            "intermediate": False,
            "selection_split": "dev",
            "final_evaluated_at_selection": False,
            "calibration": {"split": "calibration", "group_disjoint": True, "subset_sha256": "cal"},
        },
    }
    quality = {
        "split": "test",
        "evaluation_role": "test",
        "final_accuracy_target_met": True,
        "dataset_sha256": "dataset",
        "evaluated_subset_sha256": "final",
        "evaluated_records": 896,
        "checkpoint_sha256": "weights",
        "checkpoint_manifest_sha256": "manifest",
        "overall": {"questions": 896},
        "by_modality": {
            name: {"questions": size, "hard_label_questions": size, "hard_label_accuracy": 0.92}
            for name, size in (("text", 512), ("image", 384))
        },
        "by_type": dict.fromkeys(("choice", "score", "noul")),
        "by_language": dict.fromkeys(("en", "ko")),
    }
    calibration = {
        "split": "calibration",
        "evaluation_role": "calibration",
        "dataset_sha256": "dataset",
        "evaluated_subset_sha256": "cal",
        "evaluated_records": 448,
        "checkpoint_sha256": "weights",
        "calibrated_checkpoint_manifest_sha256": "manifest",
        "calibration": calibration_state,
    }
    integration = {
        "checkpoint_weights_sha256": "weights",
        "checkpoint_manifest_sha256": "manifest",
        "feature_cache": False,
        "serial_requests": True,
        "gpu": "NVIDIA GeForce RTX 4060 Ti",
        "matrix": {
            "image384-candidates4-questions1": {
                "latency_ms": {"samples": [100.0] * 40, "p95": 100.0}
            }
        },
        "http": {"local_loopback": True, "latency_ms": {"samples": [120.0] * 40, "p95": 120.0}},
        "protocol": dict.fromkeys(
            (
                "choice_id_order_pass",
                "zero_generated_tokens",
                "http_path_escape_rejected",
                "http_token_budget_rejected",
            ),
            True,
        ),
    }
    regression = {
        "evaluation_role": "regression",
        "checkpoint_sha256": "weights",
        "checkpoint_manifest_sha256": "manifest",
        "by_target_kind": {"hard": {}, "soft": {}},
        "by_family": dict.fromkeys(
            ("leaf_photo", "diagram_attributes", "text_policy", "missing_evidence")
        ),
    }
    protocol = {
        "records_sha256": "dataset",
        "final_sha256": "final",
        "split_counts": {"test": 896, "calibration": 448},
    }
    return [
        manifest,
        "weights",
        "manifest",
        quality,
        calibration,
        integration,
        regression,
        protocol,
    ]


def test_complete_evidence_passes_and_inputs_are_unchanged():
    items = evidence()
    before = copy.deepcopy(items)
    assert audit_release(*items)["passed"]
    assert items == before


@pytest.mark.parametrize("accuracy", [0.899, float("nan"), 1.1])
def test_gate_requires_raw_accuracy_in_each_modality(accuracy):
    items = evidence()
    items[3]["by_modality"]["text"].update(
        hard_label_accuracy=accuracy, accepted_expected_accuracy=1.0, coverage=0.1
    )
    assert not audit_release(*items)["passed"]


@pytest.mark.parametrize(
    "index,field,value",
    [
        (3, "evaluation_role", "regression"),
        (3, "evaluated_subset_sha256", "different-final"),
        (3, "checkpoint_manifest_sha256", "different-calibration"),
        (4, "split", "test"),
        (4, "calibrated_checkpoint_manifest_sha256", "old-model"),
        (5, "checkpoint_weights_sha256", "old-model"),
        (5, "feature_cache", True),
        (6, "by_target_kind", {"hard": {}}),
    ],
)
def test_gate_rejects_mismatched_or_incomplete_evidence(index, field, value):
    items = evidence()
    items[index][field] = value
    assert not audit_release(*items)["passed"]


@pytest.mark.parametrize(
    "samples,p95",
    [
        ([100.0] * 39, 100.0),
        ([301.0] * 40, 301.0),
        ([float("nan")] * 40, 100.0),
        ([400.0] * 40, 100.0),
    ],
)
def test_gate_checks_actual_http_timing_samples(samples, p95):
    items = evidence()
    items[5]["http"]["latency_ms"] = {"samples": samples, "p95": p95}
    assert not audit_release(*items)["passed"]


def test_smoke_or_post_final_selection_is_not_release_evidence():
    for field, value in (("smoke", True), ("final_evaluated_at_selection", True)):
        items = evidence()
        items[0]["training"][field] = value
        assert not audit_release(*items)["passed"]
