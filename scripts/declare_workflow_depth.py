"""Prospectively fix a matched earlier-layer learning study after curriculum gate failure."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path("configs/workflow-depth-study-v12.json")
    plan_path = Path("runs/workflow-v12-depth-plan/training-plan.json")
    if output.exists() or plan_path.exists():
        raise FileExistsError("the depth declaration is immutable")
    prior_path = Path("runs/workflow-v12-curriculum-selection/selection.json")
    prior = read(prior_path)
    failure = read("runs/workflow-v12-curriculum-release/progress.json")
    if (
        prior["eligible"]
        or failure["status"] != "gate_failed"
        or failure["final_predictions_started"]
    ):
        raise ValueError("the previous study must have stopped before final inference")
    profile_path = Path("runs/workflow-v12-depth-execution/results.json")
    profile = read(profile_path)
    if (
        not profile["completed"]
        or not profile["parameter_values_unchanged"]
        or profile["recommended_strategy"] != "earlier12"
        or not all(row["passed"] for row in profile["gradient_comparisons"])
    ):
        raise ValueError("the measured execution strategy has not passed")
    for path, expected in profile["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("profiled execution source changed")
    previous_config = read("configs/workflow-curriculum-study-v12.json")
    skills = Path(previous_config["skills"])
    audit = read(skills / "audit.json")
    independent = read(skills / "independent-target-audit.json")
    if not independent["passed"] or independent["data_audit_sha256"] != digest(
        skills / "audit.json"
    ):
        raise ValueError("independent skill data audit is stale")
    if not audit["passed"] or audit["calibration_or_final_used"]:
        raise ValueError("prerequisite data failed audit")
    for name, expected in audit["files_sha256"].items():
        if digest(skills / name) != expected:
            raise ValueError("skill data changed")
    data, features = Path(previous_config["records"]), Path(previous_config["features"])
    source = {}
    for line in data.open(encoding="utf-8"):
        row = json.loads(line)
        if row["split"] == "train":
            source[row["id"]] = row
    rows = read(features / "records.json")
    grouped, states = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            grouped[row["group"]].append(index)
            text_hash = hashlib.sha256(
                source[row["id"]]["request"]["state"]["text"].encode()
            ).hexdigest()
            states[text_hash].append(row["id"])
    skill_records = {
        row["id"]: row
        for row in map(
            json.loads, (skills / "train.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    skill_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for detail in map(
        json.loads, (skills / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        if detail["split"] == "train":
            skill_groups[detail["group_id"]][detail["state_sha256"]][detail["kind"]].append(detail)
    if len(skill_groups) != 960 or sum(len(value) for value in skill_groups.values()) != 1920:
        raise ValueError("expected 960 training groups with two procedural states each")
    seed, epochs = 211, 2
    orders = []
    for epoch in range(epochs):
        groups = list(grouped)
        random.Random(seed + epoch).shuffle(groups)
        order = []
        for group in groups:
            order.extend(
                {"source_id": rows[index]["id"], "cache_index": index, "extra": False}
                for index in grouped[group]
            )
            for state_hash, categories in sorted(skill_groups.get(group, {}).items()):
                controls = sorted(states[state_hash])
                offset = int(hashlib.sha256(f"{seed}:{state_hash}".encode()).hexdigest()[:8], 16)
                for index, kind in enumerate(("fact", "rule")):
                    options = sorted(categories[kind], key=lambda row: row["index"])
                    selected = options[(offset + epoch) % len(options)]
                    skill = skill_records[selected["id"]]
                    control_id = controls[(offset + epoch + index) % len(controls)]
                    if skill["request"]["state"] != source[control_id]["request"]["state"]:
                        raise ValueError("derived query differs from its original training state")
                    order.append(
                        {
                            "extra": True,
                            "skill_id": selected["id"],
                            "control_id": control_id,
                            "kind": kind,
                            "group": group,
                        }
                    )
        if Counter(item["extra"] for item in order) != {False: 11212, True: 3840}:
            raise ValueError("unexpected per-epoch training budget")
        orders.append(order)
    write(plan_path, {"seed": seed, "epochs": orders, "calibration_or_final_used": False})
    parent = Path(prior["selected"]["checkpoint"])
    if digest(parent / "head.safetensors") != prior["weights_sha256"]:
        raise ValueError("ranked parent weights changed")
    inherited = (
        "release_protocol",
        "release_protocol_sha256",
        "records",
        "records_sha256",
        "features",
        "feature_cache_manifest_sha256",
        "skills",
        "skills_audit_sha256",
        "skills_independent_audit_sha256",
        "baseline_development",
        "baseline_development_sha256",
        "fresh_calibration_design",
        "fresh_calibration_design_sha256",
        "selection",
        "data_policy",
    )
    config = {key: previous_config[key] for key in inherited}
    config.update(
        experiment="workflow-depth-study-v12",
        status="declared_before_training",
        declared_at_utc=datetime.now(timezone.utc).isoformat(),
        parent=str(parent),
        parent_weights_sha256=digest(parent / "head.safetensors"),
        parent_manifest_sha256=digest(parent / "manifest.json"),
        parent_status="Highest-ranked curriculum checkpoint, ineligible for release.",
        prior_study="configs/workflow-curriculum-study-v12.json",
        prior_study_sha256=digest("configs/workflow-curriculum-study-v12.json"),
        prior_selection=str(prior_path),
        prior_selection_sha256=digest(prior_path),
        training_plan=str(plan_path),
        training_plan_sha256=digest(plan_path),
        execution_profile=str(profile_path),
        execution_profile_sha256=digest(profile_path),
        seed=seed,
        epochs=epochs,
        learning_rate=3e-5,
        readout_lr_multiplier=0.1,
        accumulation=8,
        method="rloo",
        extra_loss_weight=1.0,
        extra_questions_per_epoch=3840,
        arms=["full_depth", "late_control"],
        earlier_learning_rate_multiplier={"full_depth": 1.0, "late_control": 0.0},
        original_adapter_layers=12,
        expanded_adapter_layers=24,
        new_adapter_initialization=(
            "Same seeded rank-8 A matrices and exactly zero B matrices in both arms."
        ),
        allocator_memory_fraction=0.85,
        checkpointing={
            "strategy": "earlier12 through 693 tokens; all24 above 693 tokens",
            "threshold_tokens": 693,
            "selection_basis": (
                "earlier12 was fastest among strategies completing all four declared inputs "
                "with identical measured logits and gradients. 693 is the longest measured "
                "input, not a proven maximum corpus shape. Full checkpointing is the declared "
                "memory fallback for longer inputs, selected before model forward."
            ),
        },
        hypothesis=(
            "Earlier language-layer adaptation may improve evidence/rule representations and "
            "their probability ordering beyond an equally scheduled late-layer continuation."
        ),
        objective=(
            "Both arms train the same original 11212 questions plus 3840 prerequisite queries "
            "per epoch, with unchanged domain weighting, RLOO+CE and Foundation retention. "
            "The sole optimization difference is the earlier adapters' learning rate."
        ),
        matched_design=(
            "Same parent, initialization seed, all 24 installed adapter layers, exact query/order, "
            "forward/backward counts, gradient clipping, AdamW parameter groups/state, warmup "
            "and cosine schedule. Earlier adapters in late_control compute gradients and "
            "optimizer state but have zero learning rate; their values must remain unchanged. "
            "One seed and two epochs per arm; no seed-robustness claim."
        ),
        inference_modules_added=0,
        generated_answer_tokens=0,
        runtime_prompt_changed=False,
        release_allowed=False,
    )
    write(output, config)
    print(
        json.dumps(
            {"configuration": str(output), "sha256": digest(output), "release_allowed": False}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
