"""Check train/dev candidate mapping and quantify the mixed target semantics on CPU."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_foundation_head import write

from veyra.candidates import candidates_for
from veyra.constants import QUESTION_TYPES
from veyra.data import read_records
from veyra.option_model import option_prompt
from veyra.transfer_learning import TransferReadout


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    output = Path("runs/workflow-v12-training-target-audit.json")
    if output.exists():
        raise FileExistsError("training target audit is immutable")
    source = Path("data/workflow-v12/records.jsonl")
    root = Path("data/features-workflow-v12")
    spec, cache = read(root / "specification.json"), read(root / "manifest.json")
    if (
        spec["records_sha256"] != digest(source)
        or spec["calibration_or_final_encoded"] is not False
    ):
        raise ValueError("expected original train/development cache")
    for name, key in (
        ("features.safetensors", "features_sha256"),
        ("records.json", "metadata_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if digest(root / name) != cache[key]:
            raise ValueError("cache fingerprint changed")
    records = {r.id: r for r in read_records(source) if r.split in {"train", "dev"}}
    rows, tensors = read(root / "records.json"), load_file(root / "features.safetensors")
    if len(rows) != len(records) or {row["id"] for row in rows} != set(records):
        raise ValueError("cache has a different observation population")
    strata = defaultdict(list)
    typed_groups = defaultdict(dict)
    maximum_difference = 0.0
    for index, row in enumerate(rows):
        record = records[row["id"]]
        name, question = next(iter(record.request.questions.items()))
        candidates = candidates_for(question)
        _, positions = option_prompt(record.request, question)
        mapping = {c.key: p for c, p in zip(candidates, positions, strict=True)}
        target = record.targets[name]
        expected = torch.zeros(16)
        for key, probability in target.items():
            expected[mapping[key]] = probability
        maximum_difference = max(
            maximum_difference, float((expected - tensors["targets"][index]).abs().max())
        )
        gold = next((tag[9:] for tag in record.tags if tag.startswith("gold_key:")), None)
        valid = torch.arange(16) < len(candidates)
        if (
            maximum_difference > 1e-6
            or int(tensors["gold"][index]) != (mapping[gold] if gold is not None else -1)
            or int(tensors["types"][index]) != QUESTION_TYPES.index(question.type)
            or not torch.equal(valid, tensors["valid"][index])
            or (row["group"], row["split"], row["family"], row["question"])
            != (record.group_id, record.split, record.family, name)
        ):
            raise ValueError(
                "cache target or metadata differs from rendered candidates: " + record.id
            )
        maximum = max(target.values())
        modes = {key for key, value in target.items() if abs(value - maximum) < 1e-8}
        strata[(record.split, row["domain"], question.type)].append(
            {
                "gold": mapping[gold] if gold is not None else -1,
                "levels": len(candidates),
                "target_max": maximum,
                "entropy": -sum(p * math.log(p) for p in target.values() if p > 0),
                "teacher": "teacher_distribution" in record.tags,
                "gold_in_modes": gold in modes if gold is not None else None,
            }
        )
        if row["domain"] == "workflow_new" and record.split == "dev":
            typed_groups[record.group_id][question.type] = {
                "index": index,
                "semantic_positions": {
                    c.description: p for c, p in zip(candidates, positions, strict=True)
                },
            }
    # This diagnosis intentionally uses the earlier readout-only checkpoint. The live backbone
    # candidates cannot be measured on an old feature cache, even if the tensor shapes match.
    checkpoint = Path(read("runs/workflow-v12-head/selection.json")["selected"]["checkpoint"])
    baseline_path = Path(read("configs/workflow-release-v12.json")["baseline"])
    state, baseline_state = (
        load_file(checkpoint / "head.safetensors"),
        load_file(baseline_path / "head.safetensors"),
    )

    def mutable(key):
        return key == "readout.weight" or key.startswith("binding_head.")

    if set(state) != set(baseline_state) or any(
        not torch.equal(value, baseline_state[key])
        for key, value in state.items()
        if not mutable(key)
    ):
        raise ValueError("cached representations do not describe this checkpoint's backbone")
    torch.set_num_threads(4)
    model = TransferReadout(read(checkpoint / "manifest.json"), state).eval()
    with torch.inference_mode():
        logits = model(tensors["states"], tensors["condition_logits"])
    disagreements, table = 0, Counter()
    meanings = ["Outcome PROCEED.", "Outcome REVIEW.", "Outcome HOLD."]
    for group, views in typed_groups.items():
        if set(views) != set(QUESTION_TYPES):
            raise ValueError("incomplete development typed group")
        predictions, correct = {}, {}
        for kind in ("choice", "score"):
            view = views[kind]
            positions = [view["semantic_positions"][meaning] for meaning in meanings]
            prediction = int(logits[view["index"], positions].argmax())
            predictions[kind] = prediction
            correct[kind] = positions[prediction] == int(tensors["gold"][view["index"]])
        disagreements += predictions["choice"] != predictions["score"]
        table[str((correct["choice"], correct["score"]))] += 1
    details = {}
    for (split, domain, kind), population in sorted(strata.items()):
        details[f"{split}/{domain}/{kind}"] = {
            "questions": len(population),
            "gold_position_counts": dict(
                sorted(Counter(row["gold"] for row in population).items())
            ),
            "candidate_count_histogram": dict(
                sorted(Counter(row["levels"] for row in population).items())
            ),
            "mean_target_max": sum(row["target_max"] for row in population) / len(population),
            "mean_target_entropy_nats": sum(row["entropy"] for row in population) / len(population),
            "teacher_distribution_questions": sum(row["teacher"] for row in population),
            "hard_gold_outside_target_modes": sum(
                row["gold_in_modes"] is False for row in population
            ),
        }
    result = {
        "passed": True,
        "records_checked": len(rows),
        "maximum_cached_target_absolute_difference": maximum_difference,
        "source_sha256": digest(Path(__file__)),
        "records_sha256": digest(source),
        "feature_cache": cache,
        "calibration_or_final_used": False,
        "model_or_policy_changed": False,
        "target_strata": details,
        "earlier_readout_only_typed_diagnostic": {
            "checkpoint": str(checkpoint),
            "weights_sha256": digest(checkpoint / "head.safetensors"),
            "manifest_sha256": digest(checkpoint / "manifest.json"),
            "unchanged_backbone_verified": True,
            "development_groups": len(typed_groups),
            "choice_score_semantic_disagreements": disagreements,
            "correctness_counts_choice_then_score": dict(table),
            "scope": (
                "Earlier unmerged readout-only model on frozen development representations; "
                "not the live continuation model."
            ),
        },
        "interpretation": (
            "Correct candidate/target alignment excludes a mapping bug. Mixed hard and teacher-"
            "distribution targets and typed-view disagreements motivate further model diagnosis; "
            "they do not by themselves identify the cause of the live model's score ranking errors."
        ),
    }
    write(output, result)
    print(
        json.dumps(
            {
                "passed": True,
                "rows": len(rows),
                "max_difference": maximum_difference,
                "earlier_typed_diagnostic": result["earlier_readout_only_typed_diagnostic"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
