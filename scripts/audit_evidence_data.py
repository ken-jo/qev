"""Independently audit generated evidence and targets; no production inference imports."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.evidence_data import policy_branches
from veyra.interventions import training_units
from veyra.policy_data import COLORS

path = Path("data/policy-v6/records.jsonl")
records = read_records(path)
original = {record.id: record for record in read_records(Path("data/policy-v3/records.jsonl"))}
assets = {}


def text_truth(expression, values):
    if " OR " in expression:
        return any(text_truth(part, values) for part in expression.split(" OR "))
    if " AND " in expression:
        return all(text_truth(part, values) for part in expression.split(" AND "))
    expression = expression.strip("() ")
    interval = re.fullmatch(r"(\d+) <= (.+?) < (\d+)", expression)
    if interval:
        low, field, high = interval.groups()
        return int(low) <= values[field] < int(high)
    field, operator, threshold = re.fullmatch(r"(.+?) (>=|<) (\d+)", expression).groups()
    return values[field] >= int(threshold) if operator == ">=" else values[field] < int(threshold)


def image_objects(asset):
    if asset in assets:
        return assets[asset]
    with Image.open(asset) as source:
        pixels = np.asarray(source.convert("RGB"))
    occupied = np.any(pixels != 255, axis=-1)
    columns = occupied.any(axis=0)
    transitions = np.diff(np.pad(columns.astype(int), (1, 1)))
    starts, stops = np.where(transitions == 1)[0], np.where(transitions == -1)[0]
    objects = []
    for start, stop in zip(starts, stops, strict=True):
        submask = occupied[:, start:stop]
        rows = np.flatnonzero(submask.any(axis=1))
        fraction = submask[rows[0] : rows[-1] + 1].mean()
        shape = "square" if fraction > 0.9 else "circle" if fraction > 0.65 else "triangle"
        colored = pixels[:, start:stop][submask]
        assert np.all(colored == colored[0])
        color = next(c for c in COLORS if tuple(colored[0]) == ImageColor.getrgb(c))
        objects.append((color, shape))
    assets[asset] = objects
    return objects


verified = 0
for record in records:
    if "evidence_intervention" not in record.tags:
        assert record == original[record.id]
        continue
    assert record.split == "train"
    parent = original[record.id.removesuffix("-evidence")]
    assert record.request.questions == parent.request.questions
    predicates, outcomes = policy_branches(record)
    if record.family.startswith("text_"):
        values = {
            field.strip(): int(value)
            for field, value in re.findall(r"([^;:]+?) = (\d+)", record.request.state.text)
        }
        truths = [text_truth(predicate, values) for predicate in predicates]
    else:
        objects = image_objects(path.parent / record.request.state.images[0].path)
        predicate = predicates[0]
        if record.family == "image_count_rule":
            truth = len(objects) >= int(re.search(r"at least (\d+)", predicate).group(1))
        else:
            observed = objects[0] if "leftmost" in predicate else objects[-1]
            probe = predicate.split()[-1]
            truth = observed[0 if record.family == "image_color_rule" else 1] == probe
        truths = [truth]
    branch = next((i for i, truth in enumerate(truths) if truth), len(predicates))
    assert f"oracle_condition_branch:{branch}" in record.tags
    answer = outcomes[branch]
    for name, question in record.request.questions.items():
        if question.type == "noul":
            probe = re.search(r"Is the policy outcome '([^']+)'\?", question.instructions).group(1)
            expected = {"false": float(answer != probe), "true": float(answer == probe)}
        else:
            expected = {
                candidate.key: float(candidate.description == answer)
                for candidate in candidates_for(question)
            }
        assert record.targets[name] == expected
    verified += 1
units = training_units([record for record in records if record.split == "train"])
report = {
    "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "verified_evidence_records": verified,
    "images_recovered_from_actual_pixels": len(assets),
    "all_parent_records_equal": True,
    "training_units": len(units),
    "training_requests_including_auxiliary": sum(map(len, units)),
    "audit": (
        "Independent numeric comparisons and pixel-derived shape/color/count; no model predictions."
    ),
}
Path("reports/evidence-v6-audit.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
