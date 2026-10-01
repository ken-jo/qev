"""Controlled dynamic-policy corpus with counterfactual rules and frozen split vocabularies."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from veyra.build_data import make_record
from veyra.data import write_records

DOMAINS = {
    "train": [
        ("duration", "users"),
        ("weight", "items"),
        ("age", "count"),
        ("delay", "size"),
        ("distance", "load"),
    ],
    "dev": [("latency", "queue"), ("length", "entries")],
    "calibration": [("height", "width"), ("depth", "slots")],
    "test": [
        ("sample index", "buffer depth"),
        ("batch age", "packet count"),
        ("signal level", "window size"),
    ],
}
LABELS = {
    "train": ["dispatch", "review", "archive", "hold", "accept", "retry"],
    "dev": ["red queue", "blue queue", "green queue", "white queue"],
    "calibration": ["north lane", "south lane", "east lane", "west lane"],
    "test": ["Lumen desk", "Quartz desk", "Sable desk", "Orchid desk"],
}
WRAPPERS = {
    "train": [
        "Apply the following policy.",
        "Use this request's routing rule.",
        "Choose according to the supplied criteria.",
    ],
    "dev": ["Determine the decision from this rule."],
    "calibration": ["Resolve this case under the stated mapping."],
    "test": [
        "Assign the observation under this previously supplied rubric.",
        "Identify the outcome required by the policy below.",
    ],
}
COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("square", "circle", "triangle")


def typed_decision(rng, instructions, labels, answer, type_name, probe):
    if type_name == "choice":
        order = rng.sample(range(len(labels)), len(labels))
        keys = [f"id{number}" for number in rng.sample(range(100, 999), len(labels))]
        criteria = {key: labels[i] for key, i in zip(keys, order, strict=True)}
        return (
            {"type": "choice", "instructions": instructions, "criteria": criteria},
            {key: float(value == answer) for key, value in criteria.items()},
        )
    if type_name == "score":
        return (
            {
                "type": "score",
                "instructions": instructions
                + " Select the ordered level whose description matches the policy outcome.",
                "criteria": labels,
            },
            {str(i): float(value == answer) for i, value in enumerate(labels)},
        )
    return (
        {"type": "noul", "instructions": instructions + f" Is the policy outcome '{probe}'?"},
        {"false": float(answer != probe), "true": float(answer == probe)},
    )


def build_policy_data(output: Path, seed: int = 20261002, scale: int = 1) -> dict:
    if scale < 1:
        raise ValueError("scale must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("policy output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir()
    source = {
        "id": "veyra-policy-generator-v3",
        "revision": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "license": "Apache-2.0",
        "url": "https://github.com/ken-jo/veyra",
    }
    sizes = {
        "train": (800 * scale, 400 * scale),
        "dev": (128, 96),
        "calibration": (128, 96),
        "test": (256, 192),
    }
    seen_images, semantic_splits, records = set(), {}, []
    for split, (text_count, image_count) in sizes.items():
        for modality, count in (("text", text_count), ("image", image_count)):
            for index in range(count):
                group = f"policy-v3-{seed}-{split}-{modality}-{index}"
                rng = random.Random(group)
                ko = (index // 4) % 4 == 0
                type_name = ("choice", "score", "noul")[
                    (index // 3 if modality == "image" else index) % 3
                ]
                labels = rng.sample(LABELS[split], 3 if index % 4 == 3 else 2)
                wrapper = rng.choice(WRAPPERS[split])
                if ko:
                    wrapper = "관측 자료에 아래 정책을 적용하고 해당하는 결과를 판단하세요."
                tags = ["dynamic_policy", "counterfactual_pair"]
                image_hash = None
                if modality == "text":
                    family = ("threshold", "conjunction", "disjunction", "ranges")[index % 4]
                    field, other = rng.choice(DOMAINS[split])
                    boundary, second_boundary = rng.randint(10, 85), rng.randint(10, 85)
                    x, y = rng.randint(0, 99), rng.randint(0, 99)
                    if index % 5 == 0:
                        x = boundary
                        tags.append("numeric_boundary")
                    state = {"text": f"Observation: {field} = {x}; {other} = {y}."}
                    first = f"{field} >= {boundary}"
                    second = f"{other} >= {second_boundary}"
                    if family == "ranges":
                        low, high = sorted(rng.sample(range(5, 95), 2))
                        predicates = [
                            f"{field} < {low}",
                            f"{low} <= {field} < {high}",
                            f"{field} >= {high}",
                        ]
                        branch = 0 if x < low else 1 if x < high else 2
                    else:
                        if family == "threshold":
                            expression, truth = first, x >= boundary
                        elif family == "conjunction":
                            expression = f"({first}) AND ({second})"
                            truth = x >= boundary and y >= second_boundary
                        else:
                            expression = f"({first}) OR ({second})"
                            truth = x >= boundary or y >= second_boundary
                        predicates = [expression, "otherwise"]
                        branch = 0 if truth else 1
                else:
                    family = ("color_rule", "shape_rule", "count_rule")[index % 3]
                    for _attempt in range(100):
                        n = rng.randint(1, 4)
                        objects = [(rng.choice(COLORS), rng.choice(SHAPES)) for _ in range(n)]
                        path = image_dir / (hashlib.sha256(group.encode()).hexdigest() + ".png")
                        with Image.new("RGB", (384, 256), "white") as image:
                            draw = ImageDraw.Draw(image)
                            for position, (color, shape) in enumerate(objects):
                                x, y = 10 + position * 94 + rng.randrange(16), rng.randint(20, 150)
                                if shape == "square":
                                    draw.rectangle((x, y, x + 60, y + 60), fill=color)
                                elif shape == "circle":
                                    draw.ellipse((x, y, x + 60, y + 60), fill=color)
                                else:
                                    draw.polygon(
                                        [(x + 30, y), (x, y + 60), (x + 60, y + 60)], fill=color
                                    )
                            image.save(path)
                        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                        if image_hash not in seen_images:
                            seen_images.add(image_hash)
                            break
                    else:
                        raise ValueError("unable to generate an independent image")
                    state = {
                        "text": "Use the image as the authoritative observation.",
                        "images": [{"path": path.relative_to(output).as_posix()}],
                    }
                    side = rng.choice(("leftmost", "rightmost"))
                    observed = objects[0] if side == "leftmost" else objects[-1]
                    if family == "color_rule":
                        probe = (
                            observed[0]
                            if index % 2
                            else rng.choice([c for c in COLORS if c != observed[0]])
                        )
                        condition, truth = (
                            f"the {side} object's color is {probe}",
                            observed[0] == probe,
                        )
                    elif family == "shape_rule":
                        probe = (
                            observed[1]
                            if index % 2
                            else rng.choice([s for s in SHAPES if s != observed[1]])
                        )
                        condition, truth = f"the {side} object is a {probe}", observed[1] == probe
                    else:
                        limit = rng.randint(1, 4)
                        condition, truth = (
                            f"the image contains at least {limit} objects",
                            n >= limit,
                        )
                    predicates, branch = [condition, "otherwise"], 0 if truth else 1
                    labels = labels[:2]
                    if index % 7 == 0:
                        wrong = next(c for c in COLORS if c != observed[0])
                        state["text"] += (
                            f" An unverified note says the {side} object is {wrong}. "
                            "The image overrides the note if they conflict."
                        )
                        tags.append("image_text_conflict")
                # Same observation, opposite mapping; one fixed output cannot solve both.
                probe_label = labels[branch]
                for version in (0, 1):
                    outcomes = labels[version:] + labels[:version]
                    rules = " ".join(
                        f"If {predicate}, return '{outcome}'."
                        if predicate != "otherwise"
                        else f"Otherwise return '{outcome}'."
                        for predicate, outcome in zip(predicates, outcomes, strict=True)
                    )
                    instructions = f"{wrapper} {rules}"
                    decision = typed_decision(
                        rng, instructions, labels, outcomes[branch], type_name, probe_label
                    )
                    signature = json.dumps(
                        {"state": state, "question": decision[0]}, sort_keys=True
                    )
                    if semantic_splits.setdefault(signature, split) != split:
                        raise ValueError("semantic task crosses splits")
                    records.append(
                        make_record(
                            group + f"-rule-{version}",
                            group,
                            split,
                            f"{modality}_{family}",
                            "ko" if ko else "en",
                            state,
                            {"decision": decision},
                            source,
                            image_hash,
                            tags + [f"modality_{modality}", f"rule_version_{version}"],
                        )
                    )
    summary = write_records(output / "records.jsonl", records)
    summary.update(
        {
            "seed": seed,
            "scale": scale,
            "protocol": "docs/improvement-protocol.md",
            "records_sha256": hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest(),
            "final_sha256": hashlib.sha256(
                "".join(r.model_dump_json() + "\n" for r in records if r.split == "test").encode()
            ).hexdigest(),
            "final_evaluated": False,
        }
    )
    (output / "protocol.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
