"""Training records and group-aware split integrity checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from veyra.candidates import candidates_for
from veyra.schema import DecisionRequest, Identifier, StrictModel


class Source(StrictModel):
    id: str
    revision: str
    license: str
    url: str
    upstream_split: str | None = None


class TrainingRecord(StrictModel):
    id: Identifier
    group_id: Identifier
    split: Literal["train", "dev", "calibration", "test"]
    family: Identifier
    language: Literal["en", "ko"]
    request: DecisionRequest
    targets: dict[str, dict[str, float]]
    source: Source
    image_sha256: str | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_targets(self) -> TrainingRecord:
        if set(self.targets) != set(self.request.questions):
            raise ValueError("targets must match all request question IDs")
        for name, question in self.request.questions.items():
            target = self.targets[name]
            if set(target) != {c.key for c in candidates_for(question)}:
                raise ValueError("target keys must exactly match candidate keys")
            if any(not math.isfinite(p) or p < 0 or p > 1 for p in target.values()):
                raise ValueError("targets must be finite probabilities")
            if not math.isclose(sum(target.values()), 1, abs_tol=1e-6):
                raise ValueError("target probabilities must sum to one")
        return self


def read_records(path: Path) -> list[TrainingRecord]:
    records = [
        TrainingRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit_records(records)
    root = path.parent.resolve()
    checked = {}
    for record in records:
        for image in record.request.state.images:
            asset = (root / image.path).resolve()
            if not asset.is_relative_to(root):
                raise ValueError("dataset image path escapes its root")
            if asset not in checked:
                with asset.open("rb") as stream:
                    checked[asset] = hashlib.file_digest(stream, "sha256").hexdigest()
            if checked[asset] != record.image_sha256:
                raise ValueError("dataset image checksum mismatch")
    return records


def audit_records(records: list[TrainingRecord]) -> dict:
    ids = set()
    groups: dict[str, str] = {}
    images: dict[str, str] = {}
    counts = {split: 0 for split in ("train", "dev", "calibration", "test")}
    sources = set()
    for record in records:
        if record.id in ids:
            raise ValueError(f"duplicate record ID: {record.id}")
        ids.add(record.id)
        previous = groups.setdefault(record.group_id, record.split)
        if previous != record.split:
            raise ValueError(f"group crosses splits: {record.group_id}")
        if record.image_sha256:
            previous_image_split = images.setdefault(record.image_sha256, record.split)
            if previous_image_split != record.split:
                raise ValueError("identical image bytes cross dataset splits")
        counts[record.split] += 1
        sources.add((record.source.id, record.source.revision, record.source.license))
    return {
        "records": len(records),
        "groups": len(groups),
        "split_counts": counts,
        "sources": [dict(zip(("id", "revision", "license"), values)) for values in sorted(sources)],
    }


def write_records(path: Path, records: list[TrainingRecord]) -> dict:
    summary = audit_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
