"""Recheck completed v13 delivery without new inference or changes to frozen criteria."""

import argparse
import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from veyra.workflow_release_gate import require_release_acceptance

ROOT = Path("runs/backbone-recovery-release-v13")
PACKAGE = Path("dist/huggingface/veyra-workflow-recovery-v13")
CHECKPOINT = Path("checkpoints/veyra-backbone-recovery-v13")
PROTOCOL = Path("configs/workflow-release-v12.json")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Delivery audits are immutable; choose a new output path")
    evidence, checks = {}, {}

    def read(path):
        path = Path(path)
        evidence[path.as_posix()] = digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def require(name, value):
        checks[name] = bool(value)
        if not checks[name]:
            raise ValueError("Delivery audit failed: " + name)

    acceptance = require_release_acceptance(ROOT / "release-acceptance.json", CHECKPOINT, PROTOCOL)
    read(ROOT / "release-acceptance.json")
    gates = read(PROTOCOL)["gates"]
    frozen = read(ROOT / "final-freeze.json")
    paired = read(ROOT / "paired.json")
    reconstructed = read(ROOT / "completion-paired-reconstruction.json")
    final = read(ROOT / "final/evaluation.json")
    photo = read(ROOT / "photo-http.json")
    contract = read(ROOT / "network-contract.json")
    verified = read(ROOT / "hf-package-verification.json")
    wheel_report = read(ROOT / "runtime-wheel.json")
    readiness = read("reports/workflow-v12/release-readiness.json")
    details = acceptance["details"]
    require("all_14_exact_model_checks", len(acceptance["checks"]) == 14)
    require("paired_bootstrap_reconstruction_exact", paired == reconstructed)
    require(
        "frozen_sources_unchanged",
        all(digest(path) == expected for path, expected in frozen["source_files"].items()),
    )
    workflow = paired["new_workflow"]
    required = gates["new_workflow"]
    require(
        "workflow_family_count", len(workflow["families"]) >= required["minimum_final_families"]
    )
    require(
        "workflow_group_count", workflow["accuracy"]["groups"] >= required["minimum_final_groups"]
    )
    require(
        "workflow_paired_improvement",
        workflow["accuracy"]["delta_95_ci"][0] > required["paired_accuracy_delta_95_lower_gt"],
    )
    require("workflow_families_disjoint", frozen["final_families_disjoint"] is True)
    uncertainty, limits = paired["uncertainty"], gates["uncertainty"]
    for name, limit_key in (
        ("nll", "nll_delta_lt"),
        ("brier", "brier_delta_lt"),
        ("matched_coverage_cost", "matched_coverage_expected_cost_delta_lt"),
    ):
        require(name + "_improves", uncertainty[name]["delta"] < limits[limit_key])
    require(
        "matched_coverage_fixed",
        uncertainty["matched_coverage_cost"]["coverage"] == limits["matched_answer_coverage"],
    )
    require(
        "paired_nll_improvement",
        uncertainty["nll"]["delta_95_ci"][1] < limits["paired_nll_delta_95_upper_lt"],
    )
    require(
        "all_uncertainty_conditions",
        set(uncertainty["condition_groups"]) == {"missing", "conflicting", "shifted_prior"},
    )
    require(
        "uncertainty_group_counts",
        all(
            n >= limits["minimum_final_groups_per_condition"]
            for n in uncertainty["condition_groups"].values()
        ),
    )
    abstention = details["uncertainty_abstention"]
    require("temperature_policy_groups_disjoint", abstention["calibration_groups_disjoint"] is True)
    require(
        "overall_final_coverage",
        abstention["overall_final"]["coverage"] >= limits["final_overall_minimum_coverage"],
    )
    require(
        "uncertain_final_abstention",
        abstention["uncertainty_final"]["abstention_rate"]
        >= limits["final_uncertainty_minimum_abstention_rate"],
    )
    populations = {
        "fitting": abstention["calibration"],
        **details["independent_policy_validation"],
        "original_group_regression": details["previous_calibration_policy_regression"],
    }
    for name, results in populations.items():
        require(name + "_all_types", set(results) == {"choice", "score", "noul"})
        for kind, result in results.items():
            require(
                name + "_" + kind,
                result["expected_error"] <= limits["calibration_expected_error_max"]
                and result["coverage"] >= limits["calibration_minimum_coverage"],
            )
            for cohort, stats in result.get("by_cohort", {}).items():
                require(
                    "fitting_" + kind + "_cohort_" + cohort,
                    stats["expected_error"] <= result["policy_fit_expected_error_max"]
                    and stats["coverage"] >= limits["calibration_minimum_coverage"],
                )
    for domain in gates["retention"]["domains"]:
        require(
            "retention_" + domain,
            details["retention_development"]["development"][domain]["delta"]
            >= -gates["retention"]["development_max_accuracy_drop"],
        )
    for modality in ("text", "image"):
        require(
            "legacy_" + modality,
            details["legacy_" + modality]["accuracy"]
            >= gates["retention"]["legacy_" + modality + "_accuracy_min"],
        )
    require(
        "official_workflow",
        details["official_workflow"]["accuracy"] >= gates["official_workflow_accuracy_min"],
    )
    latency = gates["latency"]
    require("photo_http_latency", photo["latency_ms"]["p95"] <= latency["http_p95_ms_max"])
    require("photo_http_requests", photo["requests"] == latency["photos"])
    for key in ("questions_per_request", "candidates_per_question", "warmups", "resident"):
        require("photo_http_" + key, photo["protocol"][key] == latency[key])
    require(
        "photo_http_scope",
        photo["protocol"]["photos_per_request"] == 1
        and photo["protocol"]["feature_cache"] is False
        and photo["protocol"]["serial"] is True
        and "RTX 4060 Ti" in photo["gpu"],
    )
    require(
        "photo_http_unique_observations",
        len(set(photo["protocol"]["record_ids"]))
        == photo["requests"] + photo["protocol"]["warmups"],
    )
    require(
        "network_contract",
        contract["passed"] is True
        and all(
            row["backbone_forwards"] == 1
            and row["generated_answer_tokens"] == 0
            and row["valid_probabilities"] is True
            for row in contract["results"].values()
        ),
    )
    checksums = read(PACKAGE / "checksums.json")
    actual_files = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file()}
    require("package_inventory_exact", actual_files == set(checksums) | {"checksums.json"})
    package_root = PACKAGE.resolve()
    require(
        "package_hashes",
        all(
            (package_root / name).resolve().is_relative_to(package_root)
            and digest(package_root / name) == expected
            for name, expected in checksums.items()
        ),
    )
    require(
        "package_inference_verified",
        verified["passed"] is True
        and verified["python_isolated"] is True
        and verified["verified_files"] == len(checksums),
    )
    require(
        "package_checksum_binding",
        verified["checksums_sha256"]
        == digest(PACKAGE / "checksums.json")
        == readiness["package"]["checksums_sha256"],
    )
    for name, key in (("head.safetensors", "weights_sha256"), ("manifest.json", "manifest_sha256")):
        require(
            "package_" + key,
            digest(PACKAGE / name)
            == digest(CHECKPOINT / name)
            == verified[key]
            == frozen[key]
            == readiness[key],
        )
    wheel = PACKAGE / "runtime/veyra-0.3.0a4-py3-none-any.whl"
    require(
        "wheel_hash_binding",
        digest(wheel)
        == verified["wheel_sha256"]
        == wheel_report["wheel_sha256"]
        == digest(wheel_report["wheel"]),
    )
    with zipfile.ZipFile(wheel) as archive:
        modules = [n for n in archive.namelist() if n.startswith("veyra/") and n.endswith(".py")]
        require(
            "wheel_source_bytes",
            len(modules) == wheel_report["python_modules_compared"]
            and all(archive.read(n) == (Path("src") / n).read_bytes() for n in modules),
        )
    require(
        "active_imports_from_wheel",
        all(
            Path(path).is_relative_to(wheel.resolve())
            for path in verified["modules_from_bundled_wheel"].values()
        ),
    )
    for name, row in verified["results"].items():
        require(
            "packaged_inference_" + name,
            row["backbone_forwards"] == 1 and row["response"]["usage"]["output_tokens"] == 0,
        )
        for answer in row["response"]["answers"].values():
            values = list(answer["probabilities"].values())
            require(
                "probabilities_" + name + "_" + answer["type"],
                all(math.isfinite(x) and 0 <= x <= 1 for x in values)
                and abs(math.fsum(values) - 1) < 1e-5,
            )
    require(
        "packaged_photo_all_types",
        {
            answer["type"]
            for answer in verified["results"]["photograph_three_questions"]["response"][
                "answers"
            ].values()
        }
        == {"choice", "score", "noul"},
    )
    public_count = 0
    for path in sorted(Path("reports/workflow-v13").rglob("manifest.json")):
        inventory = read(path)
        require(
            "public_manifest_" + path.parent.name,
            all(digest(path.parent / name) == expected for name, expected in inventory.items()),
        )
        public_count += len(inventory)
    diagnostic = read("runs/backbone-recovery-diagnostics-v13/recovery-final-uncertainty.json")
    require(
        "diagnostic_source",
        diagnostic["source_sha256"] == digest("scripts/diagnose_recovery_final_uncertainty.py"),
    )
    require(
        "diagnostic_inputs",
        all(digest(path) == expected for path, expected in diagnostic["evidence_files"].items()),
    )
    history = readiness["prior_progress_and_failures"]
    require("failure_history_preserved", digest(history["path"]) == history["sha256"])
    require(
        "model_card_matches_verified_package",
        Path("MODEL_CARD.md").read_bytes() == (PACKAGE / "README.md").read_bytes(),
    )
    require(
        "readiness_complete",
        readiness["release_allowed"] is True
        and readiness["hf_package_verified"] is True
        and readiness["model_acceptance_sha256"] == digest(ROOT / "release-acceptance.json")
        and readiness["package"]["verification_report_sha256"]
        == digest(ROOT / "hf-package-verification.json"),
    )
    require(
        "material_risk_disclosed",
        readiness["material_limits"]["final_accepted_expected_error"]
        == final["risk"]["actual_policy"]["expected_error"]
        and readiness["material_limits"]["uncertain_accepted_expected_error"]
        == abstention["uncertainty_final"]["expected_error"],
    )
    require(
        "hf_not_published",
        readiness["published_to_huggingface"] is False
        and verified["published_to_huggingface"] is False,
    )
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(__file__),
        "passed": True,
        "required_priorities": [1, 2],
        "scope": "Fixed empirical goals and local HF preparation; actual HF publication excluded",
        "new_model_inference": False,
        "new_training_or_criteria_changes": False,
        "checks": checks,
        "counts": {
            "model_acceptance_checks": len(acceptance["checks"]),
            "acceptance_evidence_files": len(acceptance["evidence_files"]),
            "unchanged_frozen_sources": len(frozen["source_files"]),
            "public_aggregate_files_verified": public_count,
            "package_files": len(actual_files),
            "package_checksum_entries": len(checksums),
            "wheel_python_modules_compared": len(modules),
            "active_modules_imported_by_prior_inference": len(
                verified["modules_from_bundled_wheel"]
            ),
        },
        "measured_results": readiness["measured_results"],
        "material_limits": readiness["material_limits"],
        "evidence_files": evidence,
        "published_to_huggingface": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": True, "checks": len(checks), "counts": result["counts"]}))


if __name__ == "__main__":
    main()
