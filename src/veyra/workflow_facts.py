"""Training-only condition supervision; no symbolic decision procedure at inference."""

import json
import re

import torch


def stated_probabilities(text):
    definitions = re.findall(r"^([ABCD]) means ([a-z_]+) >= (\d+)\.$", text, re.MULTILINE)
    if len(definitions) != 4 or {d[0] for d in definitions} != set("ABCD"):
        raise ValueError("expected four explicitly defined training conditions")
    observations = json.JSONDecoder().raw_decode(text.split("Observed measurements: ", 1)[1])[0]
    marker = "with these specified prior probabilities: "
    priors = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])[0] if marker in text else {}
    result = {}
    for letter, field, threshold in definitions:
        observed = observations[field]
        if observed == "unobserved":
            result[letter] = float(priors[letter])
        else:
            result[letter] = float(observed >= int(threshold))
    sensor = re.search(
        r"Two independent sensors report on condition ([ABCD])\. "
        r"Sensor X says TRUE and is correct with probability ([\d.]+)\. "
        r"Sensor Y says FALSE and is correct with probability ([\d.]+)\.",
        text,
    )
    if sensor:
        letter, positive, negative = sensor.groups()
        positive, negative = float(positive), float(negative)
        prior = result[letter]
        if letter not in priors:
            raise ValueError("sensor supervision requires an unobserved condition")
        numerator = prior * positive * (1 - negative)
        denominator = numerator + (1 - prior) * (1 - positive) * negative
        result[letter] = numerator / denominator
    if any(not 0 <= value <= 1 for value in result.values()):
        raise ValueError("invalid auxiliary target")
    return [result[letter] for letter in "ABCD"]


def auxiliary_targets(rows, records):
    records = {r.id: r for r in records if r.split in {"train", "dev"}}
    target = torch.zeros(len(rows), 4)
    mask = torch.zeros(len(rows), dtype=torch.bool)
    for index, row in enumerate(rows):
        if row["split"] not in {"train", "dev"}:
            raise ValueError("auxiliary experiment cannot inspect held-out groups")
        if row["domain"] not in {"workflow_new", "uncertainty"}:
            continue
        target[index] = torch.tensor(stated_probabilities(records[row["id"]].request.state.text))
        mask[index] = True
    return target, mask
