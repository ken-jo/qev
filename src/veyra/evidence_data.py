"""Construct controlled training counterfactuals. This module is not an inference engine."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from veyra.candidates import candidates_for
from veyra.data import TrainingRecord, read_records, write_records
from veyra.policy_data import COLORS, SHAPES


def policy_branches(record):
    question = next(iter(record.request.questions.values()))
    predicates = [
        condition
        for condition, _ in re.findall(r"If (.*?), return '([^']+)'\.", question.instructions)
    ]
    outcomes = re.findall(r"return '([^']+)'", question.instructions)
    return predicates, outcomes


def original_branch(record):
    name, question = next(iter(record.request.questions.items()))
    winner = max(candidates_for(question), key=lambda c: record.targets[name][c.key])
    if question.type == "noul":
        if winner.key != "true":
            raise ValueError("use the original true-probe record as the group anchor")
        answer = re.search(r"Is the policy outcome '([^']+)'\?", question.instructions).group(1)
    else:
        answer = winner.description
    return policy_branches(record)[1].index(answer)


def text_observation(record, branch, rng):
    """Choose fresh values satisfying a known generated branch; never parse user requests."""
    predicates, _ = policy_branches(record)
    values = {
        field.strip(): rng.randrange(100)
        for field, _ in re.findall(r"([^;:]+?) = (\d+)", record.request.state.text)
    }
    family = record.family.removeprefix("text_")
    if family == "ranges":
        field, low = re.fullmatch(r"(.+?) < (\d+)", predicates[0]).groups()
        high = int(re.fullmatch(r".+? >= (\d+)", predicates[2]).group(1))
        low = int(low)
        start, stop = ((0, low), (low, high), (high, 100))[branch]
        values[field] = rng.randrange(start, stop)
    else:
        terms = (
            re.findall(r"\((.+?) >= (\d+)\)", predicates[0])
            if family in {"conjunction", "disjunction"}
            else [re.fullmatch(r"(.+?) >= (\d+)", predicates[0]).groups()]
        )
        desired_true = branch == 0
        required = list(range(len(terms)))
        if (family == "conjunction" and not desired_true) or (
            family == "disjunction" and desired_true
        ):
            required = [rng.randrange(len(terms))]
        for index in required:
            field, boundary = terms[index]
            boundary = int(boundary)
            values[field] = (
                rng.randrange(boundary, 100) if desired_true else rng.randrange(boundary)
            )
    return (
        "Observation: " + "; ".join(f"{field} = {value}" for field, value in values.items()) + "."
    )


def image_observation(record, branch, rng, destination):
    condition = policy_branches(record)[0][0]
    positive = branch == 0
    family = record.family.removeprefix("image_")
    if family == "count_rule":
        limit = int(re.search(r"at least (\d+) objects", condition).group(1))
        count = rng.randint(limit, 5) if positive else rng.randrange(limit)
    else:
        count = rng.randint(1, 4)
    objects = [[rng.choice(COLORS), rng.choice(SHAPES)] for _ in range(count)]
    if family != "count_rule":
        index = 0 if "leftmost" in condition else -1
        if family == "color_rule":
            probe = re.search(r"color is (\w+)", condition).group(1)
            objects[index][0] = probe if positive else rng.choice([c for c in COLORS if c != probe])
        elif family == "shape_rule":
            probe = re.search(r"object is a (\w+)", condition).group(1)
            objects[index][1] = probe if positive else rng.choice([s for s in SHAPES if s != probe])
        else:
            raise ValueError("unknown controlled image family")
    with Image.new("RGB", (384, 256), "white") as image:
        draw = ImageDraw.Draw(image)
        cell = 384 // max(1, count)
        size = min(60, cell - 16)
        for index, (color, shape) in enumerate(objects):
            x = index * cell + rng.randint(4, cell - size - 4)
            y = rng.randint(20, 170)
            if shape == "square":
                draw.rectangle((x, y, x + size, y + size), fill=color)
            elif shape == "circle":
                draw.ellipse((x, y, x + size, y + size), fill=color)
            else:
                draw.polygon([(x + size // 2, y), (x, y + size), (x + size, y + size)], fill=color)
        pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        image.save(destination)
    return pixels, {"objects": objects, "branch": branch}


def build_evidence_data(parent: Path, output: Path, seed=20261003):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("evidence output must be empty")
    records = read_records(parent)
    output.mkdir(parents=True, exist_ok=True)
    # Preserve all original record contents and asset paths, including frozen final groups.
    shutil.copytree(parent.parent / "images", output / "images")
    generated_dir = output / "interventions"
    generated_dir.mkdir()
    groups = defaultdict(list)
    pixels_seen = {}
    for record in records:
        if record.split == "train":
            groups[record.group_id].append(record)
        for asset in record.request.state.images:
            path = parent.parent / asset.path
            if path not in pixels_seen:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    pixels_seen[path] = hashlib.sha256(
                        str(rgb.size).encode() + rgb.tobytes()
                    ).hexdigest()
                    rgb.close()
    forbidden_pixels = set(pixels_seen.values())
    new_records, observations = [], []
    source = {
        "id": "veyra-evidence-interventions-v6",
        "revision": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "license": "Apache-2.0",
        "url": "https://github.com/ken-jo/veyra",
    }
    for group_id, group in groups.items():
        group.sort(key=lambda r: r.id)
        anchor = group[0]
        rng = random.Random(f"{seed}-{group_id}")
        previous = original_branch(anchor)
        branch = rng.choice([i for i in range(len(policy_branches(anchor)[1])) if i != previous])
        state = anchor.request.state.model_dump()
        digest, details = None, None
        if anchor.family.startswith("text_"):
            state["text"] = text_observation(anchor, branch, rng)
        else:
            image_path = generated_dir / (hashlib.sha256(group_id.encode()).hexdigest() + ".png")
            for _ in range(100):
                pixel_digest, details = image_observation(anchor, branch, rng, image_path)
                if pixel_digest not in forbidden_pixels:
                    break
                # Empty images are intentionally shareable within train, never across splits.
                if details["objects"] == [] and pixel_digest not in set(pixels_seen.values()):
                    break
            else:
                raise ValueError("could not produce an independent training image")
            forbidden_pixels.add(pixel_digest)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            state["images"] = [{"path": image_path.relative_to(output).as_posix()}]
        for record in group:
            raw = record.model_dump()
            raw.update(
                id=record.id + "-evidence",
                group_id=group_id + "-evidence",
                source=source,
                image_sha256=digest,
                tags=record.tags + [f"oracle_condition_branch:{branch}", "evidence_intervention"],
            )
            raw["request"]["state"] = state
            for name, question in record.request.questions.items():
                answer = policy_branches(record)[1][branch]
                if question.type == "noul":
                    probe = re.search(
                        r"Is the policy outcome '([^']+)'\?", question.instructions
                    ).group(1)
                    raw["targets"][name] = {
                        "false": float(answer != probe),
                        "true": float(answer == probe),
                    }
                else:
                    raw["targets"][name] = {
                        c.key: float(c.description == answer) for c in candidates_for(question)
                    }
            new_records.append(TrainingRecord.model_validate(raw))
        observations.append(
            {
                "parent_group": group_id,
                "old_branch": previous,
                "new_branch": branch,
                "image_details": details,
            }
        )
    combined = records + new_records
    summary = write_records(output / "records.jsonl", combined)

    def frozen(rows):
        return hashlib.sha256(
            "".join(r.model_dump_json() + "\n" for r in rows if r.split == "test").encode()
        ).hexdigest()

    assert frozen(records) == frozen(combined)
    report = {
        **summary,
        "seed": seed,
        "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "new_train_records": len(new_records),
        "frozen_final_sha256": frozen(combined),
        "all_heldout_records_unchanged": all(
            a == b for a, b in zip(records, combined, strict=False)
        ),
        "observations": observations,
    }
    (output / "interventions.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return {key: value for key, value in report.items() if key != "observations"}
