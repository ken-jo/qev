"""Describe matched continuation differences after all frozen development measurements exist."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from train_foundation_head import write

from veyra.constants import QUESTION_TYPES


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_accuracy(before, after, gold, metadata, domain, seed):
    chosen = [i for i, row in enumerate(metadata) if row["domain"] == domain and gold[i] >= 0]
    if not chosen:
        raise ValueError("expected hard-labeled development domain")
    groups = {}
    transitions = {"both_correct": 0, "both_wrong": 0, "improved": 0, "regressed": 0}
    for i in chosen:
        old, new = bool(before[i] == gold[i]), bool(after[i] == gold[i])
        key = (
            "both_correct"
            if old and new
            else "both_wrong"
            if not old and not new
            else "improved"
            if new
            else "regressed"
        )
        transitions[key] += 1
        group = groups.setdefault(metadata[i]["group"], [0, 0])
        group[0] += int(new) - int(old)
        group[1] += 1
    grouped = np.asarray(list(groups.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(grouped), size=(2000, len(grouped)))
    summed = grouped[samples].sum(axis=1)
    interval = np.quantile(summed[:, 0] / summed[:, 1], [0.025, 0.975]).tolist()
    return {
        "questions": len(chosen),
        "groups": len(grouped),
        "before_accuracy": float((before[chosen] == gold[chosen]).float().mean()),
        "after_accuracy": float((after[chosen] == gold[chosen]).float().mean()),
        "accuracy_delta": float(grouped[:, 0].sum() / grouped[:, 1].sum()),
        "paired_group_bootstrap_95pct_interval": interval,
        "paired_outcomes": transitions,
    }


def parameter_difference(parent, left, right):
    if set(parent) != set(left) or set(parent) != set(right):
        raise ValueError("matched checkpoints have different tensors")
    categories = {
        "backbone_adapters": lambda key: ".lora_" in key,
        "decision_readouts": lambda key: key == "readout.weight" or key.startswith("binding_head."),
        "frozen": lambda key: (
            ".lora_" not in key and key != "readout.weight" and not key.startswith("binding_head.")
        ),
    }
    result = {}
    for name, predicate in categories.items():
        keys = [key for key in parent if predicate(key)]
        a = torch.cat([(left[key] - parent[key]).double().flatten() for key in keys])
        b = torch.cat([(right[key] - parent[key]).double().flatten() for key in keys])
        difference = a - b
        result[name] = {
            "tensors": len(keys),
            "parameters": len(a),
            "primitive_update_norm": float(a.norm()),
            "control_update_norm": float(b.norm()),
            "between_arms_norm": float(difference.norm()),
            "between_arms_max_absolute": float(difference.abs().max()),
            "nonidentical_parameters": int((difference != 0).sum()),
            "update_cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            if a.norm() > 0 and b.norm() > 0
            else None,
        }
    return result


def main():
    selection_path = Path("runs/workflow-v12-workspace-selection/selection.json")
    output = selection_path.parent / "matched-development-comparison.json"
    if output.exists():
        raise FileExistsError("descriptive matched comparison is immutable")
    config_path = Path("configs/workflow-workspace-study-v12.json")
    config, selection = read(config_path), read(selection_path)
    if selection["study_protocol_sha256"] != digest(config_path):
        raise ValueError("selection does not match the declared continuation study")
    expected = [Path(config["parent"])]
    for arm in config["execution_order"]:
        root = Path("runs/workflow-v12-workspace-study") / arm
        complete = read(root / "complete.json")
        if complete["completed"] is not True:
            raise ValueError("both declared arms must complete before comparing them")
        expected.extend(root / f"epoch-{epoch}" / "checkpoint" for epoch in (1, 2))
    candidates = selection["candidates"]
    if len(candidates) != 5 or [Path(item["checkpoint"]) for item in candidates] != expected:
        raise ValueError("comparison must include exactly the five declared candidates")
    root = Path(config["features"])
    cache = read(root / "manifest.json")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
    ):
        if digest(root / name) != cache[key]:
            raise ValueError("original feature cache changed")
    tensors, metadata = load_file(root / "features.safetensors"), read(root / "records.json")
    indices = torch.tensor([i for i, row in enumerate(metadata) if row["split"] == "dev"])
    gold, valid = tensors["gold"][indices], tensors["valid"][indices]
    rows = [metadata[i] for i in indices.tolist()]
    predictions, states = [], []
    bindings = {}
    for item, checkpoint in zip(candidates, expected, strict=True):
        for path, required in (
            (checkpoint / "head.safetensors", item["weights_sha256"]),
            (checkpoint / "manifest.json", item["manifest_sha256"]),
            (Path(item["development_logits"]), item["development_logits_sha256"]),
        ):
            if digest(path) != required:
                raise ValueError("candidate or measured logits changed: " + str(path))
            bindings[str(path)] = required
        logits = load_file(item["development_logits"])
        if not torch.equal(logits["cache_indices"], indices):
            raise ValueError("development ordering differs between candidates")
        if logits["logits"].shape != valid.shape:
            raise ValueError("development logits have the wrong shape")
        predictions.append(logits["logits"].masked_fill(~valid, -1e9).argmax(-1))
        states.append(load_file(checkpoint / "head.safetensors"))
    comparisons = []
    domains = ("workflow_new", "workflow_known", "photo_guard", "text_nli", "text_intent")
    for epoch, primitive, control in ((1, 1, 3), (2, 2, 4)):
        left, right = candidates[primitive], candidates[control]
        comparisons.append(
            {
                "epoch": epoch,
                "difference_direction": "primitive minus matched continuation control",
                "by_domain": {
                    domain: paired_accuracy(
                        predictions[control], predictions[primitive], gold, rows, domain, 191
                    )
                    for domain in domains
                },
                "by_type_coverage_under_15pct_error": {
                    kind: {
                        "primitive": left["development_risk_diagnostic"][kind][
                            "maximum_coverage_at_15pct_error"
                        ],
                        "control": right["development_risk_diagnostic"][kind][
                            "maximum_coverage_at_15pct_error"
                        ],
                    }
                    for kind in QUESTION_TYPES
                },
                "uncertainty": {
                    "primitive": left["metrics"]["by_domain"]["uncertainty"],
                    "control": right["metrics"]["by_domain"]["uncertainty"],
                },
                "parameter_differences": parameter_difference(
                    states[0], states[primitive], states[control]
                ),
            }
        )
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": digest(selection_path),
        "source_sha256": digest(Path(__file__)),
        "bindings": bindings,
        "feature_cache": cache,
        "bootstrap_seed": 191,
        "bootstrap_replicates": 2000,
        "comparisons": comparisons,
        "used_for_candidate_selection": False,
        "calibration_or_final_used": False,
        "release_allowed": False,
        "limitations": (
            "Descriptive comparisons on repeatedly inspected development groups, after the "
            "declared training/selection procedure. The percentile intervals resample observation "
            "groups, not training seeds or workflow families, and are not adjusted for multiple "
            "comparisons. One matched training seed does not establish general robustness. "
            "These are merged BF16 development measurements, not independent final evidence; "
            "parameter differences do not establish causally faithful internal reasoning."
        ),
    }
    write(output, result)
    print(json.dumps({"output": str(output), "epochs_compared": len(comparisons)}), flush=True)


if __name__ == "__main__":
    main()
