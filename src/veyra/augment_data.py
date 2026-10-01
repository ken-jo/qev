"""Request-context augmentation, preserving source groups and original test records."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

from veyra.build_data import diagram_records, text_records
from veyra.data import TrainingRecord, read_records, write_records


def paraphrase(question: dict, rng: random.Random, held_out: bool = False) -> dict:
    result = dict(question)
    if held_out:
        prefixes = [
            "Evaluate this task using the supplied evidence: ",
            "Review the observations and resolve this query: ",
        ]
    else:
        prefixes = [
            "",
            "Using the evidence, decide: ",
            "Answer this question: ",
            "For this request, determine the answer. ",
            "주어진 자료에 근거해 판단하세요. ",
        ]
    instructions = result["instructions"]
    replacements = [
        ("What is the color", "Which color describes"),
        ("leftmost object", "object on the far left"),
        ("Select an action under this policy.", "Choose the action required by the rules."),
        ("Does the duration meet or exceed the limit?", "Is the duration at least the limit?"),
        ("How many flags apply?", "Determine the number of satisfied conditions."),
        (
            "Which labeled leaf condition best matches this photograph?",
            "Which description fits the leaf in this image?",
        ),
        ("Does the photograph match the condition", "Does this leaf have the condition"),
        ("Rate the number of visible objects.", "Count the objects shown in the image."),
        ("Apply the supplied review-level mapping.", "Select the level using this mapping."),
    ]
    if not held_out:
        for original, alternative in replacements:
            if rng.random() < 0.5:
                instructions = instructions.replace(original, alternative)
    result["instructions"] = rng.choice(prefixes) + instructions
    return result


def variants(record: TrainingRecord, seed: int) -> list[TrainingRecord]:
    """All variants of an image/observation remain in its original split."""
    rng = random.Random(f"{seed}:{record.id}")
    results = [record]
    names = list(record.request.questions)
    for index, size in enumerate((1, rng.randint(1, len(names)))):
        selected = rng.sample(names, size)
        raw = record.model_dump()
        raw["id"] += f"-context-{index}"
        raw["tags"] += ["single_question" if size == 1 else "variable_question_count"]
        if index:
            raw["tags"].append("paraphrased")
        raw["request"]["questions"] = {
            name: paraphrase(raw["request"]["questions"][name], rng)
            if index
            else raw["request"]["questions"][name]
            for name in selected
        }
        raw["targets"] = {name: raw["targets"][name] for name in selected}
        results.append(TrainingRecord.model_validate(raw))
    return results


def augment_dataset(source: Path, output: Path, seed: int = 20261001) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("augmented dataset output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    originals = read_records(source)
    original_ids = {record.id for record in originals}
    # Keep image bytes unchanged. Copy only validated, root-confined referenced assets.
    for record in originals:
        for image in record.request.state.images:
            destination = output / image.path
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source.parent / image.path, destination)
    expanded = []
    for record in originals + text_records(768, seed):
        if record.split == "test":
            raw = record.model_dump()
            raw["tags"].append("legacy_regression" if record.id in original_ids else "fresh_policy")
            if record.id not in original_ids:
                rng = random.Random(record.id)
                selected = rng.sample(list(raw["request"]["questions"]), rng.randint(1, 3))
                raw["request"]["questions"] = {
                    name: paraphrase(raw["request"]["questions"][name], rng, held_out=True)
                    for name in selected
                }
                raw["targets"] = {name: raw["targets"][name] for name in selected}
            expanded.append(TrainingRecord.model_validate(raw))
        else:
            expanded.extend(variants(record, seed))
    # Fresh challenge images use a new seed; no overlapping byte hashes are retained.
    known_hashes = {r.image_sha256 for r in expanded if r.image_sha256}
    for record in diagram_records(output, 160, seed + 9000):
        if record.image_sha256 and record.image_sha256 in known_hashes:
            continue
        raw = record.model_dump()
        raw["split"] = "test"
        raw["tags"].append("fresh_diagram")
        rng = random.Random(record.id)
        names = list(raw["request"]["questions"])
        selected = rng.sample(names, rng.randint(1, len(names)))
        raw["request"]["questions"] = {
            name: paraphrase(raw["request"]["questions"][name], rng, held_out=True)
            for name in selected
        }
        raw["targets"] = {name: raw["targets"][name] for name in selected}
        expanded.append(TrainingRecord.model_validate(raw))
    report = write_records(output / "records.jsonl", expanded)
    report.update(
        {
            "augmentation_seed": seed,
            "parent_dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "augmentation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "test_usage": "Legacy test is a regression set after v1 diagnostics. Fresh tags were "
            "created before v2 training and excluded from training/selection/calibration.",
        }
    )
    (output / "augmentation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
