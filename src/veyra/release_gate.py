"""Check that published quality and timing evidence belongs to one selected checkpoint."""

from __future__ import annotations

import hashlib
import math

import numpy as np


def evidence_hashes(data: bytes) -> dict[str, str]:
    """Retain exact archive bytes and the LF representation published by Git."""
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "lf_sha256": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest(),
    }


def evidence_matches(data: bytes, reference: dict) -> bool:
    hashes = evidence_hashes(data)
    return hashes["sha256"] == reference.get("sha256") or (
        bool(reference.get("lf_sha256")) and hashes["lf_sha256"] == reference["lf_sha256"]
    )


def _timing_pass(timing: dict, target_ms: float = 300) -> bool:
    samples = timing.get("samples", [])
    if len(samples) < 40 or any(not math.isfinite(x) or x <= 0 for x in samples):
        return False
    measured = float(np.percentile(samples, 95))
    return measured <= target_ms and math.isclose(timing.get("p95", -1), measured, rel_tol=1e-9)


def audit_release(
    manifest: dict,
    weights_sha256: str,
    manifest_sha256: str,
    quality: dict,
    calibration: dict,
    integration: dict,
    regression: dict,
    protocol: dict,
) -> dict:
    training = manifest.get("training", {})
    modality = quality.get("by_modality", {})
    checks = {
        "trained_selected_checkpoint": (
            training.get("completed") is True
            and training.get("optimizer_steps", 0) > 0
            and not training.get("smoke", False)
            and training.get("intermediate") is False
            and training.get("selection_split") == "dev"
            and training.get("final_evaluated_at_selection") is False
        ),
        "checkpoint_weights_verified": manifest.get("weights_sha256") == weights_sha256,
        "independent_final_role": (
            quality.get("split") == quality.get("evaluation_role") == "test"
            and quality.get("final_accuracy_target_met") is True
        ),
        "frozen_final_dataset": (
            quality.get("dataset_sha256") == protocol.get("records_sha256")
            and quality.get("evaluated_subset_sha256") == protocol.get("final_sha256")
            and quality.get("evaluated_records") == protocol.get("split_counts", {}).get("test")
            and bool(protocol.get("final_sha256"))
        ),
        "each_modality_at_least_90_before_abstention": (
            set(modality) == {"text", "image"}
            and all(
                value.get("hard_label_questions", 0) == value.get("questions")
                and value.get("questions", 0) > 0
                and 0.9 <= value.get("hard_label_accuracy", -1) <= 1
                for value in modality.values()
            )
            and sum(value.get("questions", 0) for value in modality.values())
            == quality.get("overall", {}).get("questions")
            == quality.get("evaluated_records")
        ),
        "same_final_checkpoint": (
            quality.get("checkpoint_sha256") == weights_sha256
            and quality.get("checkpoint_manifest_sha256") == manifest_sha256
        ),
        "calibration_separate_from_final": (
            calibration.get("split") == calibration.get("evaluation_role") == "calibration"
            and calibration.get("dataset_sha256") == protocol.get("records_sha256")
            and calibration.get("evaluated_records")
            == protocol.get("split_counts", {}).get("calibration")
            and training.get("calibration", {}).get("split") == "calibration"
            and training.get("calibration", {}).get("group_disjoint") is True
            and calibration.get("evaluated_subset_sha256")
            == training.get("calibration", {}).get("subset_sha256")
            and bool(calibration.get("evaluated_subset_sha256"))
            and calibration.get("checkpoint_sha256") == weights_sha256
            and calibration.get("calibrated_checkpoint_manifest_sha256") == manifest_sha256
            and calibration.get("calibration") == manifest.get("calibration")
        ),
        "same_integration_checkpoint": (
            integration.get("checkpoint_weights_sha256") == weights_sha256
            and integration.get("checkpoint_manifest_sha256") == manifest_sha256
        ),
        "uncached_serial_rtx4060ti_measurement": (
            integration.get("feature_cache") is False
            and integration.get("serial_requests") is True
            and "RTX 4060 Ti" in integration.get("gpu", "")
        ),
        "one_image_four_candidates_p95_at_most_300_ms": _timing_pass(
            integration.get("matrix", {})
            .get("image384-candidates4-questions1", {})
            .get("latency_ms", {})
        ),
        "actual_http_p95_at_most_300_ms": (
            integration.get("http", {}).get("local_loopback") is True
            and _timing_pass(integration.get("http", {}).get("latency_ms", {}))
        ),
        "dynamic_api_protocol": all(
            integration.get("protocol", {}).get(name) is True
            for name in (
                "choice_id_order_pass",
                "zero_generated_tokens",
                "http_path_escape_rejected",
                "http_token_budget_rejected",
            )
        ),
        "same_checkpoint_legacy_and_uncertainty_report": (
            regression.get("evaluation_role") == "regression"
            and regression.get("checkpoint_sha256") == weights_sha256
            and regression.get("checkpoint_manifest_sha256") == manifest_sha256
            and {"leaf_photo", "diagram_attributes", "text_policy", "missing_evidence"}
            <= set(regression.get("by_family", {}))
            and {"hard", "soft"} <= set(regression.get("by_target_kind", {}))
        ),
        "type_and_language_slices_reported": (
            {"choice", "score", "noul"} <= set(quality.get("by_type", {}))
            and {"en", "ko"} <= set(quality.get("by_language", {}))
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint_weights_sha256": weights_sha256,
        "checkpoint_manifest_sha256": manifest_sha256,
        "scope": "Frozen controlled policy final set, legacy regression and resident serial API",
        "limitations": (
            "A passing audit does not establish general real-world policy accuracy, "
            "fully Korean policy understanding, concurrent-load latency or OOD calibration."
        ),
    }
