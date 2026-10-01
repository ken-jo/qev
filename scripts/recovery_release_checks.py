"""Recompute recovery selection and bind its prospective data and historical sources."""

import hashlib
import json
import math
from pathlib import Path

import torch
from run_backbone_recovery import metrics
from safetensors.torch import load_file
from study_readout_recovery import decide
from workflow_depth_common import layer_number

from veyra.data import read_records

DESIGN = Path("configs/workflow-backbone-recovery-v13.json")
MEMORY_REPAIR = Path("configs/workflow-backbone-memory-repair-v13.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def study_root():
    repair = read(MEMORY_REPAIR)
    require(
        repair["study_design_sha256"] == digest(DESIGN)
        and digest(repair["retry_source"]) == repair["retry_source_sha256"]
        and repair["prospective_model_selection"] is True
        and repair["new_final_or_validation_outputs_used"] is False,
        "training memory repair changed the study",
    )
    for path, sha in repair["preserved_evidence"].items():
        require(digest(path) == sha, "memory failure or probe evidence changed")
    require(
        read("runs/backbone-recovery-v13/execution-failure.json")["completed"] is False
        and read("runs/backbone-recovery-memory-probe-v13.json")["passed"] is True,
        "memory repair no longer binds its failure and feasibility measurement",
    )
    return Path(repair["retry_root"])


def require(condition, message):
    if not condition:
        raise ValueError("Recovery evidence rejected: " + message)


def equal(left, right):
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and set(left) == set(right)
            and all(equal(left[k], right[k]) for k in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, bool) or left is None:
        return left is right
    if isinstance(left, (int, float)):
        return (
            isinstance(right, (int, float))
            and math.isfinite(left)
            and math.isfinite(right)
            and (abs(left - right) <= 1e-6)
        )
    return left == right


def bind_source(path, expected, evidence):
    """Only an explicit later runtime-version/gate transition may archive these two modules."""
    if digest(path) == expected:
        evidence[str(path)] = expected
        return
    normalized = Path(path).resolve().relative_to(Path.cwd()).as_posix()
    require(
        normalized in {"src/veyra/__init__.py", "src/veyra/workflow_release_gate.py"},
        "executed source changed: " + str(path),
    )
    transition_path = Path("runs/backbone-recovery-release-v13/runtime-transition.json")
    require(transition_path.is_file(), "runtime source change has no archived transition")
    transition = read(transition_path)
    change = transition["source_changes"][normalized]
    require(
        change["before_sha256"] == expected
        and digest(change["archive"]) == expected
        and change["after_sha256"] == digest(path),
        "runtime transition does not preserve source",
    )
    evidence[change["archive"]] = expected
    evidence[str(path)] = digest(path)
    evidence[str(transition_path)] = digest(transition_path)


def collect_study():
    torch.set_num_threads(4)
    config = read(DESIGN)
    root = study_root()
    report = read(root / "selection.json")
    protocol = read(root / "protocol.json")
    complete = read(root / "training-complete.json")
    require(
        report["eligible"] is True and report["final_or_calibration_used"] is False,
        "no eligible merged development candidate",
    )
    require(
        report["design_sha256"] == protocol["design_sha256"] == digest(DESIGN)
        and report["release_protocol_sha256"]
        == config["release_protocol_sha256"]
        == digest(config["release_protocol"]),
        "prospective design changed",
    )
    require(
        complete["completed"] is True
        and complete["questions"] == 9129
        and complete["source_files"] == protocol["source_files"]
        and complete["steps"] == sum(math.ceil(n / 8) for n in protocol["segment_questions"]),
        "incomplete or altered training schedule",
    )
    repair = read(MEMORY_REPAIR)
    original = read(Path(repair["original_failed_root"]) / "protocol.json")
    require(
        protocol["memory_repair_sha256"] == digest(MEMORY_REPAIR)
        and report["memory_repair_sha256"] == digest(MEMORY_REPAIR)
        and protocol["training_order_sha256"] == original["training_order_sha256"]
        and protocol["segment_questions"] == original["segment_questions"],
        "memory repair changed training order or candidate boundaries",
    )
    evidence = dict(repair["preserved_evidence"])
    evidence[str(MEMORY_REPAIR)] = digest(MEMORY_REPAIR)
    for path, sha in protocol["source_files"].items():
        bind_source(path, sha, evidence)
    for path in (
        DESIGN,
        root / "protocol.json",
        root / "training-complete.json",
        root / "parameter-plan.json",
        root / "selection.json",
    ):
        evidence[str(path)] = digest(path)
    parent = Path(config["parent"])
    require(
        digest(parent / "head.safetensors") == config["parent_weights_sha256"]
        and digest(parent / "manifest.json") == config["parent_manifest_sha256"],
        "recovery parent changed",
    )
    parent_manifest = read(parent / "manifest.json")
    parent_state = load_file(parent / "head.safetensors")
    for path in (parent / "head.safetensors", parent / "manifest.json"):
        evidence[str(path)] = digest(path)
    cache = Path(config["features"])
    require(
        digest(cache / "manifest.json") == config["features_manifest_sha256"],
        "feature completion record changed",
    )
    cache_spec = read(cache / "specification.json")
    require(cache_spec["calibration_or_final_encoded"] is False, "holdout features in optimization")
    for path, sha in cache_spec["source_files"].items():
        bind_source(path, sha, evidence)
    for split in ("train", "dev"):
        manifest = read(cache / split / "manifest.json")
        require(
            digest(cache / "specification.json") == manifest["specification_sha256"],
            "cache specification changed",
        )
        for filename, key in (
            ("features.safetensors", "features_sha256"),
            ("records.json", "metadata_sha256"),
        ):
            path = cache / split / filename
            require(digest(path) == manifest[key], "cache values changed")
            evidence[str(path)] = digest(path)
        evidence[str(cache / split / "manifest.json")] = digest(cache / split / "manifest.json")
    for filename in ("specification.json", "manifest.json"):
        evidence[str(cache / filename)] = digest(cache / filename)
    tensors, rows = load_file(cache / "dev/features.safetensors"), read(cache / "dev/records.json")
    train_rows = read(cache / "train/records.json")
    require(
        len(rows) == 5102
        and all(row["split"] == "dev" for row in rows)
        and len(train_rows) == 9129
        and all(row["split"] == "train" for row in train_rows)
        and not {r["group"] for r in rows} & {r["group"] for r in train_rows},
        "training/development identities changed or overlap",
    )
    for spec in config["source_datasets"].values():
        require(digest(spec["path"]) == spec["sha256"], "source data changed")
        evidence[spec["path"]] = spec["sha256"]
    reference_path = Path("runs/readout-recovery-v13/interpolation-protocol.json")
    baseline_path = Path("runs/workflow-v12-workspace-selection/merged-baseline-dev.json")
    parent_metrics = read(reference_path)["parent_development"]
    baseline = read(baseline_path)["metrics"]
    for path in (
        reference_path,
        baseline_path,
        Path(config["prior_failure_selection"]),
        Path(config["prior_readout_study"]),
    ):
        evidence[str(path)] = digest(path)
    require(
        read(config["prior_failure_selection"])["eligible"] is False
        and digest(config["prior_failure_selection"]) == config["prior_failure_selection_sha256"],
        "head recovery failure changed",
    )
    require(len(report["candidates"]) == 3, "candidate count changed")
    recomputed = []
    for result, fraction in zip(
        report["candidates"], config["checkpoint_group_fractions"], strict=True
    ):
        checkpoint = root / f"fraction-{fraction:g}" / "checkpoint"
        require(
            Path(result["checkpoint"]) == checkpoint
            and digest(checkpoint / "head.safetensors") == result["weights_sha256"]
            and digest(checkpoint / "manifest.json") == result["manifest_sha256"],
            "candidate identity changed",
        )
        manifest = read(checkpoint / "manifest.json")
        state = load_file(checkpoint / "head.safetensors")
        require(
            set(state) == set(parent_state)
            and manifest["adaptation"] == parent_manifest["adaptation"]
            and manifest["calibration"] == parent_manifest["calibration"]
            and manifest["backbone"] == parent_manifest["backbone"]
            and manifest["training"]["backbone_recovery_design_sha256"] == digest(DESIGN),
            "model structure or inherited calibration changed",
        )
        for key, value in state.items():
            require(torch.isfinite(value).all().item(), "nonfinite parameter")
            may_change = (
                key == "readout.weight"
                or key.startswith("binding_head.")
                or (".lora_" in key and layer_number(key) in config["trainable_adapter_layers"])
            )
            if not may_change:
                require(torch.equal(value, parent_state[key]), "frozen parameter changed: " + key)
        logits_path = checkpoint.parent / "merged-development.safetensors"
        require(
            digest(logits_path) == result["merged_development_logits_sha256"]
            and result["merged_bf16_deployment"] is True
            and result["final_or_calibration_used"] is False,
            "development output changed",
        )
        logits = load_file(logits_path)["logits"]
        require(
            tuple(logits.shape) == (5102, 16) and torch.isfinite(logits).all().item(),
            "incomplete development output",
        )
        actual = metrics(logits, tensors, rows, parent_manifest["calibration"]["temperatures"])
        movement = sum(float((state[k] - parent_state[k]).square().sum()) for k in state)
        checks, rank = decide(actual, parent_metrics, baseline, config, movement)
        require(
            equal(actual, result["metrics"])
            and checks == result["checks"]
            and equal(rank, result["selection"]),
            "development metrics or selection do not reconstruct",
        )
        recomputed.append(result)
        for path in (
            checkpoint / "head.safetensors",
            checkpoint / "manifest.json",
            logits_path,
            checkpoint.parent / "development.json",
        ):
            evidence[str(path)] = digest(path)
    selected = max(recomputed, key=lambda r: tuple(r["selection"]))
    require(
        selected == report["selected"] and selected["selection"][0] is True,
        "frozen selection differs from declared ranking",
    )
    return {
        "selected": selected,
        "evidence_files": evidence,
        "passed": True,
        "design_sha256": digest(DESIGN),
        "original_release_thresholds_unchanged": True,
    }


def collect_data():
    design_path = Path("configs/workflow-recovery-release-v13.json")
    design = read(design_path)
    completed_path = Path("runs/workflow-recovery-final-data-v13/complete.json")
    complete = read(completed_path)
    require(
        complete["completed"] is True and complete["model_outputs_used"] is False,
        "fresh population construction incomplete",
    )
    evidence = {str(design_path): digest(design_path), str(completed_path): digest(completed_path)}
    populations = {}
    specs = [
        ("validation-1", design["validation_populations"][0], "calibration", 2644, 2040),
        ("validation-2", design["validation_populations"][1], "calibration", 2644, 2040),
        (
            "final",
            {
                "design": design["final_data_design"],
                "design_sha256": design["final_data_design_sha256"],
            },
            "test",
            3432,
            1920,
        ),
    ]
    for role, spec, split, count, procedural in specs:
        require(
            digest(spec["design"]) == spec["design_sha256"], "prospective population design changed"
        )
        data_design = read(spec["design"])
        root = Path(data_design["output"])
        for path, expected in complete["outputs"][role].items():
            require(digest(path) == expected, "completed fresh population changed")
            evidence[path] = expected
        audit = read(root / "independent-target-audit.json")
        protocol = read(root / "protocol.json")
        require(
            audit["passed"] is True
            and audit["model_inference_used"] is False
            and audit["fresh_policy_targets_recomputed"] == procedural
            and audit["maximum_target_absolute_error"] <= 1e-7
            and protocol["model_outputs_used"] is False
            and digest(root / "records.jsonl")
            == audit["records_sha256"]
            == protocol["records_sha256"],
            "fresh target audit failed",
        )
        records = [r for r in read_records(root / "records.jsonl") if r.split == split]
        require(len(records) == count, "wrong fresh population size")
        require(
            all(
                r.source.upstream_split == "test"
                for r in records
                if r.source.id == "PolyAI/banking77"
            ),
            "BANKING77 pool repair provenance lost",
        )
        populations[role] = {
            "records": str(root / "records.jsonl"),
            "questions": count,
            "groups": sorted({r.group_id for r in records}),
        }
        evidence[spec["design"]] = spec["design_sha256"]
    fitting = [
        r for r in read_records(Path(design["calibration_records"])) if r.split == "calibration"
    ]
    groups = [set(pop["groups"]) for pop in populations.values()] + [{r.group_id for r in fitting}]
    require(
        all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :]),
        "fitting, validation or final observation overlap",
    )
    require(
        digest(design["calibration_records"]) == design["calibration_records_sha256"],
        "explicitly reused fitting data changed",
    )
    evidence[design["calibration_records"]] = design["calibration_records_sha256"]
    for repair in complete["repairs"]:
        evidence[repair] = digest(repair)
        item = read(repair)
        for path, sha in item["repaired_sources"].items():
            require(digest(path) == sha, "data repair source changed")
            evidence[path] = sha
    return {
        "passed": True,
        "populations": populations,
        "evidence_files": evidence,
        "final_inference_used": False,
        "release_allowed": False,
    }
