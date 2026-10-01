"""Align prior and current workflow predictions without fitting or selecting a model."""

import hashlib
import json
from pathlib import Path

from compare_foundation_predictions import compare, summarize
from evaluate_laya_reference import official_records

from veyra.decision_metrics import observation, report


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_root = Path("runs/laya-comparison-20260930")
    root = Path("runs/foundation-v11-evaluation")
    output = root / "typed-regression-audit.json"
    standardized = root / "typed-baseline-standardized.json"
    if output.exists() or standardized.exists():
        raise FileExistsError("regression audits are immutable")
    protocol = read(old_root / "protocol.json")
    if digest(old_root / "test.jsonl") != protocol["jsonl_sha256"]:
        raise ValueError("original dataset changed")
    checkpoint = Path("checkpoints/veyra-dynamic-v2")
    if (
        digest(checkpoint / "head.safetensors") != protocol["checkpoint_weights_sha256"]
        or digest(checkpoint / "manifest.json") != protocol["checkpoint_manifest_sha256"]
    ):
        raise ValueError("historical baseline checkpoint changed")
    records = {r.id: r for r in official_records(old_root / "test.jsonl")}
    old, tokens = {}, {}
    for line in (old_root / "veyra-predictions.jsonl").read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        identity = value["id"] + ":" + value["question"]
        record = records[identity]
        name = value["question"]
        if "gold_key:" + value["gold_label"] not in record.tags:
            raise ValueError("gold label differs between evaluations")
        if any(
            abs(value["gold_probabilities"][key] - target) > 1e-10
            for key, target in record.targets[name].items()
        ):
            raise ValueError("target distribution differs between evaluations")
        winner = max(value["probabilities"], key=value["probabilities"].get)
        if value["prediction"] != winner:
            raise ValueError("historical prediction was not probability argmax")
        row = observation(record, name, value["probabilities"], abstained=value["abstained"])
        if row["correct"] != value["correct"]:
            raise ValueError("historical correctness differs")
        old[(identity, name)] = row
        tokens[(identity, name)] = value["input_tokens"]
    new = {}
    new_path = root / "typed-regression/predictions.jsonl"
    new_report = read(new_path.with_name("evaluation.json"))
    if digest(new_path) != new_report["predictions_sha256"]:
        raise ValueError("new predictions changed")
    for line in new_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        new[(row["id"], row["question"])] = row
    paired = compare(old, new)
    if len(old) != 2000 or len(new) != 2000:
        raise ValueError("workflow comparison must cover all 2000 questions")
    old_report = {"protocol": protocol, **report(list(old.values()), bootstrap=True)}
    old_report["source_predictions_sha256"] = digest(old_root / "veyra-predictions.jsonl")
    old_report["metric_note"] = (
        "Recomputed aggregate metrics from preserved predictions, no new inference; "
        "score MAE now compares ordinal expectations, matching the current evaluator"
    )
    result = {
        "protocol": {
            "source_sha256": digest(Path(__file__)),
            "original_source_sha256": digest(old_root / "evaluate_veyra.py"),
            "old_predictions_sha256": old_report["source_predictions_sha256"],
            "new_predictions_sha256": digest(new_path),
            "scope": "Post-evaluation regression diagnosis only, no tuning or reselection",
            "input_contract": (
                "Both use original state JSON as text and one original typed question per call; "
                "gold labels and complete target distributions matched for all 2000 questions"
            ),
        },
        "paired": paired,
        "flips": {
            "wrong_to_correct": sum(not a["correct"] and new[k]["correct"] for k, a in old.items()),
            "correct_to_wrong": sum(a["correct"] and not new[k]["correct"] for k, a in old.items()),
        },
        "by_prior_input_length": {
            name: summarize([(old[k], new[k]) for k in old if low < tokens[k] <= high])
            for name, low, high in (
                ("up_to_256", 0, 256),
                ("257_to_512", 256, 512),
                ("over_512", 512, 100000),
            )
            if any(low < value <= high for value in tokens.values())
        },
        "interpretation": (
            "The observed workflow regression is not a gold-label or hard/soft metric mismatch. "
            "Transfer across domains/formats remains limited. The causal contribution of training "
            "distribution, instruction length and adaptation is not isolated by this audit."
        ),
    }
    for path, value in ((output, result), (standardized, old_report)):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "old": old_report["overall"]["accuracy"],
                "new": new_report["overall"]["accuracy"],
                "flips": result["flips"],
                "paired": paired["overall"],
            }
        )
    )


if __name__ == "__main__":
    main()
