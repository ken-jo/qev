"""Require exact-model evidence for the mandatory workflow release priorities."""

import hashlib
import json
from pathlib import Path

REQUIRED_CHECKS = (
    "workflow_generalization",
    "uncertainty_probability",
    "uncertainty_abstention",
    "retention_development",
    "legacy_text",
    "legacy_image",
    "official_workflow",
    "runtime",
    "single_network",
    "provenance",
)
EXTENSION_CHECKS = {
    "workflow_recovery_v13": (
        "recovery_study_provenance", "fresh_data_provenance",
        "independent_policy_validation", "previous_calibration_policy_regression",
    ),
    "workflow_workspace_v12": (
        "workspace_study_provenance",
        "previous_calibration_policy_regression",
    ),
    "workflow_curriculum_v12": (
        "curriculum_study_provenance",
        "previous_calibration_policy_regression",
    ),
    "workflow_depth_v12": (
        "depth_study_provenance",
        "previous_calibration_policy_regression",
    ),
    "workflow_depth_policy_v12": (
        "depth_study_provenance",
        "conservative_policy_study_provenance",
        "independent_policy_validation",
        "previous_calibration_policy_regression",
    ),
    "workflow_cohort_policy_v12": (
        "depth_study_provenance",
        "cohort_policy_study_provenance",
        "independent_policy_validation",
        "previous_calibration_policy_regression",
    ),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_release_acceptance(report_path, checkpoint, protocol_path):
    if not Path(report_path).is_file():
        raise ValueError("Release blocked: priorities 1 and 2 require a passing acceptance report")
    result = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if result.get("release_allowed") is not True or result.get("required_priorities") != [1, 2]:
        raise ValueError("Release blocked: both mandatory roadmap priorities must pass")
    if any(result.get("checks", {}).get(key) is not True for key in REQUIRED_CHECKS):
        raise ValueError("Release blocked: missing or failing mandatory evidence")
    if result.get("release_protocol_sha256") != digest(protocol_path):
        raise ValueError("Release blocked: acceptance used another protocol")
    for name, key in (("head.safetensors", "weights_sha256"), ("manifest.json", "manifest_sha256")):
        if result.get(key) != digest(Path(checkpoint) / name):
            raise ValueError("Release blocked: acceptance describes another checkpoint")
    manifest = json.loads((Path(checkpoint) / "manifest.json").read_text(encoding="utf-8"))
    extensions = manifest.get("training", {}).get("release_extensions", {})
    if result.get("release_extensions", {}) != extensions:
        raise ValueError("Release blocked: study extensions do not match the calibrated model")
    for extension in extensions:
        if extension not in EXTENSION_CHECKS or any(
            result.get("checks", {}).get(key) is not True for key in EXTENSION_CHECKS[extension]
        ):
            raise ValueError("Release blocked: missing or failed continuation-study evidence")
    if any(value is not True for value in result.get("checks", {}).values()):
        raise ValueError("Release blocked: an additional mandatory check failed")
    evidence = result.get("evidence_files", {})
    if not evidence:
        raise ValueError("Release blocked: source evidence is missing")
    for path, expected in evidence.items():
        if digest(path) != expected:
            raise ValueError("Release blocked: evidence fingerprint mismatch: " + path)
    return result
