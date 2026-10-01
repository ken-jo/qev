"""Reproducible, licensed starter corpus spanning photos, diagrams, and text policies."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

from veyra.data import TrainingRecord, write_records

BEANS_ID = "AI-Lab-Makerere/beans"
BEANS_REVISION = "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
BEAN_LABELS = {
    "healthy": "healthy",
    "angular_leaf_spot": "angular leaf spot",
    "bean_rust": "bean rust",
}


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return (
        "train"
        if bucket < 70
        else "dev"
        if bucket < 80
        else "calibration"
        if bucket < 90
        else "test"
    )


def choice(rng: random.Random, instructions: str, meanings: list[str], correct: str) -> tuple:
    rng.shuffle(meanings)
    keys = rng.sample(range(1000, 9999), len(meanings))
    criteria = {f"option_{key}": value for key, value in zip(keys, meanings, strict=True)}
    return (
        {"type": "choice", "instructions": instructions, "criteria": criteria},
        {key: float(value == correct) for key, value in criteria.items()},
    )


def noul(instructions: str, truth: bool | None) -> tuple:
    probability = 0.5 if truth is None else float(truth)
    return (
        {"type": "noul", "instructions": instructions},
        {"false": 1.0 - probability, "true": probability},
    )


def score(instructions: str, criteria: list[str], target: int) -> tuple:
    return (
        {"type": "score", "instructions": instructions, "criteria": criteria},
        {str(i): float(i == target) for i in range(len(criteria))},
    )


def make_record(
    record_id: str,
    group: str,
    split: str,
    family: str,
    language: str,
    state: dict,
    decisions: dict,
    source: dict,
    image_hash: str | None = None,
    tags: list[str] | None = None,
) -> TrainingRecord:
    return TrainingRecord.model_validate(
        {
            "id": record_id,
            "group_id": group,
            "split": split,
            "family": family,
            "language": language,
            "request": {"state": state, "questions": {k: v[0] for k, v in decisions.items()}},
            "targets": {k: v[1] for k, v in decisions.items()},
            "source": source,
            "image_sha256": image_hash,
            "tags": tags or [],
        }
    )


def original_source() -> dict:
    return {
        "id": "veyra-original-generator",
        "revision": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "license": "Apache-2.0",
        "url": "https://github.com/ken-jo/veyra",
    }


def diagram_records(root: Path, count: int, seed: int) -> list[TrainingRecord]:
    source = original_source()
    records = []
    images = root / "images" / "diagrams"
    images.mkdir(parents=True, exist_ok=True)
    seen_images = set()
    for index in range(count):
        rng = random.Random(seed + index)
        group = f"diagram-{seed}-{index}"
        split = split_for(group)
        ko = index % 3 == 0
        held_out = split == "test" and index % 2 == 0
        colors = ["red", "blue", "green", "yellow"] + (["purple"] if held_out else [])
        shapes = ["square", "circle", "triangle"]
        n = rng.randint(1, 4)
        objects = [(rng.choice(colors), rng.choice(shapes)) for _ in range(n)]
        path = images / f"{group}.png"
        with Image.new("RGB", (384, 256), "white") as image:
            draw = ImageDraw.Draw(image)
            for i, (color, shape) in enumerate(objects):
                x = 15 + i * 94 + rng.randint(0, 6)
                y = 85 + rng.randint(-15, 15)
                bounds = (x, y, x + 65, y + 65)
                if shape == "square":
                    draw.rectangle(bounds, fill=color)
                elif shape == "circle":
                    draw.ellipse(bounds, fill=color)
                else:
                    draw.polygon([(x + 32, y), (x + 65, y + 65), (x, y + 65)], fill=color)
            image.save(path)
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if image_hash in seen_images:
            continue
        seen_images.add(image_hash)
        color, shape = objects[0]
        probe = rng.choice(shapes)
        criteria = rng.sample(
            [c for c in colors if c != color], rng.randint(1, len(colors) - 1)
        ) + [color]
        prompt = (
            "가장 왼쪽 물체의 색은 무엇인가요?"
            if ko
            else "What is the color of the leftmost object?"
        )
        if held_out and not ko:
            prompt = "Identify the hue of the object nearest the left edge."
        text = "Use the picture as evidence."
        tags = ["unseen_template", "unseen_color_allowed"] if held_out else []
        if index % 5 == 0:
            wrong = next(c for c in colors if c != color)
            text += (
                f" An unverified note claims the leftmost object is {wrong}. "
                "Trust the image if they disagree."
            )
            tags.append("image_text_conflict")
        decisions = {
            "color": choice(rng, prompt, criteria, color),
            "shape": noul(
                f"가장 왼쪽 물체의 모양이 {probe}인가요?"
                if ko
                else f"Is the leftmost object a {probe}?",
                shape == probe,
            ),
            "count": score(
                "보이는 물체의 개수를 평가하세요." if ko else "Rate the number of visible objects.",
                [f"Exactly {i} visible objects." for i in range(5)],
                n,
            ),
        }
        records.append(
            make_record(
                group,
                group,
                split,
                "diagram_attributes",
                "ko" if ko else "en",
                {"text": text, "images": [{"path": path.relative_to(root).as_posix()}]},
                decisions,
                source,
                image_hash,
                tags,
            )
        )
        if index % 10 == 0:
            missing = {
                "color": choice(
                    rng,
                    prompt,
                    ["red", "blue", "Not enough visual evidence to decide."],
                    "Not enough visual evidence to decide.",
                ),
                "shape": noul(f"Is the unobserved object a {probe}?", None),
            }
            records.append(
                make_record(
                    group + "-missing",
                    group,
                    split,
                    "missing_evidence",
                    "ko" if ko else "en",
                    {"text": "No image or description of the object was supplied."},
                    missing,
                    source,
                    tags=["missing_evidence", "soft_target"],
                )
            )
    return records


def text_records(count: int, seed: int) -> list[TrainingRecord]:
    records = []
    source = original_source()
    for index in range(count):
        rng = random.Random(seed + 10000 + index)
        group = f"text-{seed}-{index}"
        split = split_for(group)
        ko = index % 3 == 0
        duration, affected = rng.randint(0, 90), rng.randint(0, 50)
        limit, user_limit = rng.randint(10, 70), rng.randint(5, 40)
        level = int(duration >= limit) + int(affected >= user_limit)
        escalate = level == 2
        state = {"text": f"Outage duration: {duration} minutes. Affected users: {affected}."}
        policy = f"Flag A: duration >= {limit} minutes. Flag B: affected users >= {user_limit}."
        meaning = (
            "Escalate: both flags apply." if escalate else "Monitor: fewer than two flags apply."
        )
        instructions = (
            "주어진 정책에 따라 조치를 선택하세요. "
            if ko
            else "Select an action under this policy. "
        )
        if split == "test":
            instructions = (
                "Apply these new routing criteria. "
                if not ko
                else "다음 기준으로 담당 조치를 결정하세요. "
            )
        decisions = {
            "action": choice(
                rng,
                instructions + policy,
                ["Escalate: both flags apply.", "Monitor: fewer than two flags apply."],
                meaning,
            ),
            "condition": noul(
                (
                    "장애 시간이 기준 이상인가요? "
                    if ko
                    else "Does the duration meet or exceed the limit? "
                )
                + f"Limit: {limit} minutes.",
                duration >= limit,
            ),
            "flags": score(
                ("충족되는 조건의 개수를 평가하세요. " if ko else "How many flags apply? ")
                + policy,
                ["No flags apply.", "Exactly one flag applies.", "Both flags apply."],
                level,
            ),
        }
        records.append(
            make_record(
                group,
                group,
                split,
                "text_policy",
                "ko" if ko else "en",
                state,
                decisions,
                source,
                tags=["unseen_template"] if split == "test" else [],
            )
        )
    return records


def bean_records(root: Path, cache_dir: str, seed: int) -> list[TrainingRecord]:
    from huggingface_hub import hf_hub_download

    records = []
    images = root / "images" / "beans"
    images.mkdir(parents=True, exist_ok=True)
    seen = set()
    for upstream in ("train", "validation", "test"):
        archive_path = hf_hub_download(
            BEANS_ID,
            repo_type="dataset",
            revision=BEANS_REVISION,
            filename=f"data/{upstream}.zip",
            cache_dir=cache_dir,
        )
        with ZipFile(archive_path) as archive:
            for name in sorted(archive.namelist()):
                if not name.lower().endswith(".jpg"):
                    continue
                label = BEAN_LABELS[Path(name).parent.name]
                if archive.getinfo(name).file_size > 20 * 1024 * 1024:
                    raise ValueError("unexpected oversized source image")
                raw = archive.read(name)
                image_hash = hashlib.sha256(raw).hexdigest()
                # Byte-identical duplicates are assigned only to their first source split.
                if image_hash in seen:
                    continue
                seen.add(image_hash)
                path = images / f"{image_hash}.jpg"
                path.write_bytes(raw)
                rng = random.Random(int(image_hash[:16], 16) + seed)
                split = upstream
                if upstream == "validation":
                    split = "dev" if int(image_hash[:8], 16) % 2 else "calibration"
                group = "bean-" + image_hash[:32]
                ko = int(image_hash[:8], 16) % 3 == 0
                labels = list(BEAN_LABELS.values())
                proposition = rng.choice(labels)
                order = rng.sample(labels, len(labels))
                rubric = [
                    f"Level {i}: the labeled condition is {value}." for i, value in enumerate(order)
                ]
                question = (
                    "사진의 잎에 해당하는 상태를 선택하세요."
                    if ko
                    else "Which labeled leaf condition best matches this photograph?"
                )
                if upstream == "test" and not ko:
                    question = "Assign the photographed leaf to the matching description."
                decisions = {
                    "condition": choice(rng, question, labels.copy(), label),
                    "matches": noul(
                        f"사진의 잎 상태가 {proposition}에 해당하나요?"
                        if ko
                        else f"Does the photograph match the condition '{proposition}'?",
                        label == proposition,
                    ),
                    "policy_level": score(
                        "Apply the supplied review-level mapping. "
                        "The levels specify a review policy.",
                        rubric,
                        order.index(label),
                    ),
                }
                source = {
                    "id": BEANS_ID,
                    "revision": BEANS_REVISION,
                    "license": "MIT",
                    "url": "https://github.com/AI-Lab-Makerere/ibean",
                    "upstream_split": upstream,
                }
                records.append(
                    make_record(
                        group,
                        group,
                        split,
                        "leaf_photo",
                        "ko" if ko else "en",
                        {
                            "text": "A field photograph of a bean leaf.",
                            "images": [{"path": path.relative_to(root).as_posix()}],
                        },
                        decisions,
                        source,
                        image_hash,
                        ["unseen_template"] if split == "test" else [],
                    )
                )
    return records


def build_dataset(
    output: Path,
    scenes: int = 512,
    text_cases: int = 512,
    seed: int = 20260930,
    include_beans: bool = True,
    cache_dir: str = ".cache/huggingface",
) -> dict:
    if scenes < 0 or text_cases < 0:
        raise ValueError("dataset counts must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("dataset output must be empty")
    records = diagram_records(output, scenes, seed) + text_records(text_cases, seed)
    if include_beans:
        records += bean_records(output, cache_dir, seed)
    summary = write_records(output / "records.jsonl", records)
    summary["seed"] = seed
    return summary
