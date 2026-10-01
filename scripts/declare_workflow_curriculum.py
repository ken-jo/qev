"""Freeze a matched prerequisite-curriculum study before either training arm starts."""

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
    output = Path("configs/workflow-curriculum-study-v12.json")
    plan_path = Path("runs/workflow-v12-curriculum-plan/training-plan.json")
    if output.exists() or plan_path.exists():
        raise FileExistsError("the curriculum declaration is immutable")
    failure = read("runs/workflow-v12-workspace-evaluation/progress.json")
    diagnostic = Path("runs/workflow-v12-skill-diagnostic")
    if failure["status"] != "gate_failed" or failure["final_predictions_started"]:
        raise ValueError("previous study did not stop before final inference")
    if not read(diagnostic / "complete.json")["completed"]:
        raise ValueError("finish the prospective prerequisite diagnosis first")
    skills = Path("data/workflow-v12-skills")
    audit = read(skills / "audit.json")
    independent = read(skills / "independent-target-audit.json")
    if not independent["passed"] or independent["data_audit_sha256"] != digest(
        skills / "audit.json"
    ):
        raise ValueError("independent skill target audit is missing or stale")
    if not audit["passed"] or audit["calibration_or_final_used"]:
        raise ValueError("prerequisite data failed audit")
    for name, expected in audit["files_sha256"].items():
        if digest(skills / name) != expected:
            raise ValueError("skill curriculum data changed")
    data = Path("data/workflow-v12/records.jsonl")
    features = Path("data/features-workflow-v12")
    source = {}
    for line in data.open(encoding="utf-8"):
        row = json.loads(line)
        if row["split"] == "train":
            source[row["id"]] = row
    rows = read(features / "records.json")
    grouped, states = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] == "train":
            grouped[row["group"]].append(i)
            text_hash = hashlib.sha256(
                source[row["id"]]["request"]["state"]["text"].encode()
            ).hexdigest()
            states[text_hash].append(row["id"])
    skill_records = {
        r["id"]: r for r in map(json.loads, (skills / "train.jsonl").read_text().splitlines())
    }
    skill_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for detail in map(json.loads, (skills / "annotations.jsonl").read_text().splitlines()):
        if detail["split"] == "train":
            skill_groups[detail["group_id"]][detail["state_sha256"]][detail["kind"]].append(detail)
    if len(skill_groups) != 960 or sum(len(x) for x in skill_groups.values()) != 1920:
        raise ValueError("expected 960 training groups with two distinct states each")
    seed, epochs = 197, 2
    orders = []
    for epoch in range(epochs):
        group_ids = list(grouped)
        random.Random(seed + epoch).shuffle(group_ids)
        order = []
        for group in group_ids:
            order.extend(
                {"source_id": rows[i]["id"], "cache_index": i, "extra": False}
                for i in grouped[group]
            )
            for state_hash, categories in sorted(skill_groups.get(group, {}).items()):
                controls = sorted(states[state_hash])
                offset = int(hashlib.sha256(f"{seed}:{state_hash}".encode()).hexdigest()[:8], 16)
                for index, kind in enumerate(("fact", "rule")):
                    options = sorted(categories[kind], key=lambda r: r["index"])
                    selected = options[(offset + epoch) % len(options)]
                    skill = skill_records[selected["id"]]
                    control_id = controls[(offset + epoch + index) % len(controls)]
                    control = source[control_id]
                    if skill["request"]["state"] != control["request"]["state"]:
                        raise ValueError("matched extra queries must see the same input state")
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
    parent = Path("runs/workflow-v12-workspace-study/primitive/epoch-2/checkpoint")
    config = {
        "experiment": "workflow-curriculum-study-v12",
        "status": "declared_before_training",
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Separate study after matched workspace selection failed; "
            "previous artifacts stay immutable."
        ),
        "release_protocol": "configs/workflow-release-v12.json",
        "release_protocol_sha256": digest("configs/workflow-release-v12.json"),
        "parent": str(parent),
        "parent_weights_sha256": digest(parent / "head.safetensors"),
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "records": str(data),
        "records_sha256": digest(data),
        "features": str(features),
        "feature_cache_manifest_sha256": digest(features / "manifest.json"),
        "skills": str(skills),
        "skills_audit_sha256": digest(skills / "audit.json"),
        "skills_independent_audit_sha256": digest(skills / "independent-target-audit.json"),
        "training_plan": str(plan_path),
        "training_plan_sha256": digest(plan_path),
        "prior_selection_sha256": digest("runs/workflow-v12-workspace-selection/selection.json"),
        "prerequisite_diagnostic_sha256": digest(diagnostic / "results.json"),
        "baseline_development": "runs/workflow-v12-workspace-selection/merged-baseline-dev.json",
        "baseline_development_sha256": digest(
            "runs/workflow-v12-workspace-selection/merged-baseline-dev.json"
        ),
        "fresh_calibration_design": "configs/workflow-workspace-calibration-v12.json",
        "fresh_calibration_design_sha256": digest(
            "configs/workflow-workspace-calibration-v12.json"
        ),
        "seed": seed,
        "epochs": epochs,
        "learning_rate": 3e-5,
        "readout_lr_multiplier": 0.1,
        "accumulation": 8,
        "method": "rloo",
        "extra_loss_weight": 1.0,
        "extra_questions_per_epoch": 3840,
        "arms": ["skill_curriculum", "outcome_replay"],
        "objective": (
            "Both arms repeat all 11212 original training questions per epoch with unchanged "
            "domain weighting and frozen-Foundation replay. Each adds 3840 questions from the "
            "same original procedural states. Curriculum asks one fact and one rule condition "
            "per state; control repeats original outcome queries on those same states. All "
            "extras use the same RLOO+CE objective and unit weight. No auxiliary decoder."
        ),
        "matched_design": (
            "Same parent, initialization seed, original examples, state order, number of "
            "forward/backward passes, optimizer updates and schedule. Extra query wording, "
            "candidate count and type differ; exact token/FLOP budgets are not matched. "
            "One training seed, two epochs per arm; no seed-robustness claim."
        ),
        "selection": (
            "Parent plus both epochs of both arms, all with merged BF16 inference on original "
            "development only. Original retention, novel-workflow and probability eligibility, "
            "plus each type's maximum threshold coverage >=60% under <=15% expected error. "
            "Rank eligible candidates by original known/new macro accuracy, uncertainty NLL "
            "and cost at 80% coverage. Skill diagnostics are never selection criteria."
        ),
        "data_policy": (
            "Original train/development/calibration/final records stay unchanged. Derived skill "
            "questions inherit their source split/group; only training skills are optimized. "
            "The uninspected fresh calibration design may be used only after candidate "
            "selection; then the unchanged-policy original-calibration regression is mandatory. "
            "Freeze before original final inference. Any inspected final requires fresh groups "
            "for a later performance claim."
        ),
        "inference_modules_added": 0,
        "generated_answer_tokens": 0,
        "runtime_prompt_changed": False,
        "release_allowed": False,
    }
    write(output, config)
    print(
        json.dumps(
            {
                "declaration": str(output),
                "sha256": digest(output),
                "questions_per_epoch_per_arm": len(orders[0]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
