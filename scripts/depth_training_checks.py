"""Audit the fixed depth-study population, initialization and actual saved adapter updates."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file
from workflow_depth_common import layer_number


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Depth study rejected: " + message)


def audit_training():
    config_path = Path("configs/workflow-depth-study-v12.json")
    config = read(config_path)
    require(config["arms"] == ["full_depth", "late_control"] and config["epochs"] == 2, "design")
    require(
        config["earlier_learning_rate_multiplier"] == {"full_depth": 1.0, "late_control": 0.0},
        "earlier-layer comparison changed",
    )
    files = {str(config_path): digest(config_path)}
    for key, path in (
        ("records", config["records"]),
        ("training_plan", config["training_plan"]),
        ("execution_profile", config["execution_profile"]),
        ("prior_study", config["prior_study"]),
        ("prior_selection", config["prior_selection"]),
        ("release_protocol", config["release_protocol"]),
    ):
        require(digest(path) == config[key + "_sha256"], "input changed: " + str(path))
        files[str(path)] = digest(path)
    train = {}
    for line in Path(config["records"]).open(encoding="utf-8"):
        row = json.loads(line)
        if row["split"] == "train":
            train[row["id"]] = row
    skills = {
        row["id"]: row
        for row in map(
            json.loads,
            (Path(config["skills"]) / "train.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    plan = read(config["training_plan"])
    require(plan["seed"] == config["seed"] and len(plan["epochs"]) == config["epochs"], "schedule")
    for order in plan["epochs"]:
        primary = [row for row in order if not row["extra"]]
        extra = [row for row in order if row["extra"]]
        require(
            Counter(row["source_id"] for row in primary) == Counter({key: 1 for key in train})
            and len(primary) == 11212
            and len(extra) == 3840,
            "original training questions must each appear exactly once per epoch",
        )
        states = defaultdict(Counter)
        for item in extra:
            skill, original = skills[item["skill_id"]], train[item["control_id"]]
            require(
                skill["split"] == original["split"] == "train"
                and skill["group_id"] == original["group_id"] == item["group"]
                and skill["request"]["state"] == original["request"]["state"],
                "prerequisite query is outside its original training state/group",
            )
            state = json.dumps(skill["request"]["state"], sort_keys=True)
            states[state][item["kind"]] += 1
        require(
            len(states) == 1920
            and all(value == Counter({"fact": 1, "rule": 1}) for value in states.values()),
            "each training state must have exactly one fact and one rule extra",
        )
    parent_path = Path(config["parent"])
    parent_manifest = read(parent_path / "manifest.json")
    parent_state = load_file(parent_path / "head.safetensors")
    require(
        digest(parent_path / "head.safetensors") == config["parent_weights_sha256"]
        and digest(parent_path / "manifest.json") == config["parent_manifest_sha256"],
        "parent identity",
    )
    added_shapes = {}
    for name, tensor in parent_state.items():
        if ".lora_" in name:
            number = layer_number(name)
            require(12 <= number < 24, "parent depth")
            added = name.replace(
                f"language_model.layers.{number}.", f"language_model.layers.{number - 12}."
            )
            added_shapes[added] = tensor.shape
    expected_shapes = {name: tensor.shape for name, tensor in parent_state.items()} | added_shapes
    root = Path("runs/workflow-v12-depth-study")
    completion = read(root / "complete.json")
    require(
        completion["completed"] and completion["configuration_sha256"] == digest(config_path),
        "both arms must finish",
    )
    sources, initializations = {}, []
    control_state = None
    for arm in config["arms"]:
        arm_root = root / arm
        protocol, initialization = (
            read(arm_root / "protocol.json"),
            read(arm_root / "initialization.json"),
        )
        initializations.append(initialization)
        require(
            protocol["configuration_sha256"]
            == initialization["configuration_sha256"]
            == digest(config_path)
            and protocol["training_plan_sha256"] == config["training_plan_sha256"]
            and protocol["calibration_or_final_used"] is False
            and datetime.fromisoformat(config["declared_at_utc"])
            <= datetime.fromisoformat(protocol["started_at_utc"]),
            "training was not prospectively declared",
        )
        sources.update(protocol["source_sha256"])
        for epoch in range(1, config["epochs"] + 1):
            checkpoint = arm_root / f"epoch-{epoch}" / "checkpoint"
            manifest = read(checkpoint / "manifest.json")
            state = load_file(checkpoint / "head.safetensors")
            training = manifest["training"]
            for key in (
                "backbone",
                "encoder",
                "decision_views",
                "reasoning_slots",
                "binding_head",
                "calibration",
            ):
                require(manifest[key] == parent_manifest[key], "inference contract changed: " + key)
            require(
                manifest["adaptation"] == {**parent_manifest["adaptation"], "layers": 24}
                and training["protocol_sha256"] == digest(arm_root / "protocol.json")
                and training["initialization_sha256"] == digest(arm_root / "initialization.json")
                and training["optimizer_steps"] == epoch * math.ceil(15052 / config["accumulation"])
                and training["epochs"] == epoch
                and training["earlier_adapters_updated"] == (arm == "full_depth")
                and manifest["weights_sha256"] == digest(checkpoint / "head.safetensors"),
                "checkpoint differs from its declared update schedule",
            )
            require(
                {name: value.shape for name, value in state.items()} == expected_shapes
                and torch.equal(
                    state["condition_readout.weight"], parent_state["condition_readout.weight"]
                ),
                "saved parameter layout or frozen condition head changed",
            )
            counts = training["memory_policy_counts"]
            require(counts["earlier12"] + counts["all24"] == epoch * 15052, "forward budget")
            earlier = {name: state[name] for name in added_shapes}
            b_nonzero = any(
                bool(torch.count_nonzero(value))
                for name, value in earlier.items()
                if name.endswith(".lora_b")
            )
            require(b_nonzero == (arm == "full_depth"), "actual earlier adapter updates disagree")
            unchanged = (
                training["parameter_group_sha256"]["earlier_adapters"]
                == initialization["parameter_group_sha256"]["earlier_adapters"]
            )
            require(unchanged == (arm == "late_control"), "earlier adapter fingerprint disagrees")
            if arm == "late_control":
                if control_state is not None:
                    require(
                        all(
                            torch.equal(control_state[key], value) for key, value in earlier.items()
                        ),
                        "control earlier tensors changed across epochs",
                    )
                control_state = earlier
            for filename in ("head.safetensors", "manifest.json"):
                files[str(checkpoint / filename)] = digest(checkpoint / filename)
        for filename in (
            "protocol.json",
            "initialization.json",
            "first-gradient.json",
            "complete.json",
            "history.json",
        ):
            files[str(arm_root / filename)] = digest(arm_root / filename)
    for key in (
        "trainable_parameter_sha256",
        "parameter_group_sha256",
        "parameter_count_by_group",
        "expansion",
        "configuration_sha256",
    ):
        require(
            initializations[0][key] == initializations[1][key], "matched initialization changed"
        )
    for path, expected in sources.items():
        require(digest(path) == expected, "training source changed: " + path)
    sources[str(Path(__file__))] = digest(__file__)
    files.update(sources)
    return {
        "passed": True,
        "evidence_files": files,
        "source_files": sources,
        "calibration_or_final_used": False,
    }
