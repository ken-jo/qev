"""Predeclared fresh groups and vocabularies after the v3 final set was retired."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from veyra.data import TrainingRecord, read_records, write_records
from veyra.image_inputs import fingerprint_context, pixel_fingerprints
from veyra.packing import pack_request
from veyra.policy_data import DOMAINS, LABELS, build_policy_data

SEEDS = (20261004, 20261005, 20261006, 20261007)
DOMAINS_V8 = {
    "train": [
        ("period", "units"),
        ("mass", "requests"),
        ("elapsed", "capacity"),
        ("range", "demand"),
        ("length", "volume"),
    ],
    "dev": [("transit rating", "input volume"), ("pending age", "supply size")],
    "calibration": [("cycle length", "slot usage"), ("phase delay", "reserve count")],
    "test": [
        ("crystal reading", "turbine rating"),
        ("beacon strength", "interval depth"),
        ("module tension", "segment height"),
    ],
}
LABELS_V8 = {
    "train": [
        "route Aster",
        "route Birch",
        "route Cinder",
        "route Dune",
        "route Ember",
        "route Flint",
    ],
    "dev": ["Maple station", "Cobalt station", "Hazel station", "Poppy station"],
    "calibration": ["Silver bay", "Copper bay", "Bronze bay", "Golden bay"],
    "test": ["Nimbus office", "Violet office", "Cedar office", "Harbor office"],
}
WRAPPERS_V8 = {
    "train": "Apply the specified mapping to the evidence.",
    "dev": "Determine which outcome the observation implies under this specification.",
    "calibration": "Use the stated conditions to identify this observation's assigned result.",
    "test": "Read the supplied evidence and select the outcome dictated by the rules.",
}
WRAPPERS_KO_V8 = {
    "train": "주어진 관측값에 지정된 규칙을 적용하세요.",
    "dev": "이 규칙에 따라 관측 자료의 결과를 고르세요.",
    "calibration": "제시된 조건을 확인하고 관측에 해당하는 결과를 정하세요.",
    "test": "관측 내용과 규칙을 대조하여 요구되는 결과를 선택하세요.",
}


def remap_record(record: TrainingRecord, prefix: str, revision: str) -> TrainingRecord:
    replacements = dict(zip(LABELS[record.split], LABELS_V8[record.split], strict=True))
    for old, new in zip(DOMAINS[record.split], DOMAINS_V8[record.split], strict=True):
        replacements.update(zip(old, new, strict=True))
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(x) for x in sorted(replacements, key=len, reverse=True))
        + r")(?!\w)"
    )

    def rewrite(text):
        return pattern.sub(lambda match: replacements[match.group()], text)

    raw = record.model_dump()
    raw["source"] = {
        "id": "veyra-policy-refresh-v8",
        "revision": revision,
        "license": "Apache-2.0",
        "url": "https://github.com/ken-jo/veyra",
    }
    raw["request"]["state"]["text"] = rewrite(raw["request"]["state"]["text"])
    for image in raw["request"]["state"]["images"]:
        image["path"] = prefix + "/" + image["path"]
    wrappers = WRAPPERS_KO_V8 if record.language == "ko" else WRAPPERS_V8
    for question in raw["request"]["questions"].values():
        instruction = rewrite(question["instructions"])
        first_rule = instruction.find("If ")
        if first_rule < 0:
            raise ValueError("expected a controlled policy")
        question["instructions"] = wrappers[record.split] + " " + instruction[first_rule:]
        if question["type"] == "choice":
            question["criteria"] = {
                key: rewrite(value) for key, value in question["criteria"].items()
            }
        elif question["type"] == "score":
            question["criteria"] = [rewrite(value) for value in question["criteria"]]
    return TrainingRecord.model_validate(raw)


def build_refresh(output: Path, exclude: list[Path]) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refresh output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    revision = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    seen_pixels, seen_tasks, pixel_cache = set(), set(), {}

    def fingerprints(record, root):
        pixels = pixel_fingerprints(record.request.state.images, root, pixel_cache)
        task = hashlib.sha256(
            (pack_request(record.request).text + fingerprint_context(pixels)).encode()
        ).hexdigest()
        return pixels, task

    excluded_sources = []
    for path in exclude:
        old = read_records(path)
        excluded_sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "records": len(old),
                "splits": "all previous observations, not just training",
            }
        )
        for record in old:
            pixels, task = fingerprints(record, path.parent)
            seen_pixels.update(pixels)
            seen_tasks.add(task)
    records, rejected = [], []
    for seed in SEEDS:
        prefix = f"raw-{seed}"
        raw_root = output / prefix
        build_policy_data(raw_root, seed=seed, scale=1)
        groups = defaultdict(list)
        for source in read_records(raw_root / "records.jsonl"):
            record = remap_record(source, prefix, revision)
            groups[record.group_id].append(record)
        for name, group in groups.items():
            pairs = [fingerprints(record, output) for record in group]
            duplicate = any(
                seen_pixels.intersection(pixels) or task in seen_tasks for pixels, task in pairs
            )
            if duplicate:
                rejected.append(
                    {"group": name, "split": group[0].split, "reason": "duplicate observation"}
                )
                continue
            records.extend(group)
            for pixels, task in pairs:
                seen_pixels.update(pixels)
                seen_tasks.add(task)
    summary = write_records(output / "records.jsonl", records)
    report = {
        **summary,
        "seeds": SEEDS,
        "selection_by_model_results": False,
        "source_revision": revision,
        "excluded_sources": excluded_sources,
        "duplicate_groups_removed": rejected,
        "records_sha256": hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest(),
        "split_sha256": {
            split: hashlib.sha256(
                "".join(r.model_dump_json() + "\n" for r in records if r.split == split).encode()
            ).hexdigest()
            for split in ("train", "dev", "calibration", "test")
        },
        "final_evaluated": False,
        "scope": "Fresh controlled-policy groups and disjoint split vocabularies; synthetic images",
    }
    report["final_sha256"] = report["split_sha256"]["test"]
    (output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
