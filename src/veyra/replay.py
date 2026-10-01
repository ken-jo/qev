"""Balanced replay of training observations; original split membership is preserved."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from veyra.data import TrainingRecord
from veyra.packing import pack_request
from veyra.schema import DecisionRequest


def audit_replay(primary, primary_root: Path, replay, replay_root: Path) -> dict:
    """Reject shared held-out groups, canonical questions, and decoded image pixels."""
    heldout = [record for record in primary if record.split != "train"]
    if any(record.split != "train" for record in replay):
        raise ValueError("replay only accepts training records")
    heldout_groups = {record.group_id for record in heldout}
    if any(record.group_id in heldout_groups for record in replay):
        raise ValueError("replay group overlaps a held-out observation")
    cache = {}

    def fingerprints(record, root):
        pixel = None
        if record.request.state.images:
            path = (root / record.request.state.images[0].path).resolve()
            if path not in cache:
                with Image.open(path) as source:
                    rgb = source.convert("RGB")
                    cache[path] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    rgb.close()
            pixel = cache[path]
        tasks = []
        for question in record.request.questions.values():
            request = DecisionRequest(state=record.request.state, questions={"q": question})
            tasks.append(
                hashlib.sha256((pack_request(request).text + str(pixel)).encode()).hexdigest()
            )
        return pixel, tasks

    forbidden_pixels, forbidden_tasks = set(), set()
    for record in heldout:
        pixel, tasks = fingerprints(record, primary_root)
        if pixel:
            forbidden_pixels.add(pixel)
        forbidden_tasks.update(tasks)
    for record in replay:
        pixel, tasks = fingerprints(record, replay_root)
        if pixel and pixel in forbidden_pixels:
            raise ValueError("replay pixels overlap a held-out observation")
        if forbidden_tasks.intersection(tasks):
            raise ValueError("replay question overlaps a held-out request")
    return {"heldout_records_checked": len(heldout), "replay_train_records_checked": len(replay)}


def replay_pool(records: list[TrainingRecord]) -> dict[str, list[TrainingRecord]]:
    if any(record.split != "train" for record in records):
        raise ValueError("replay only accepts training records")
    groups = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
    families = defaultdict(list)
    for group in groups.values():
        # Context variants must not multiply an observation's sampling weight.
        record = min(group, key=lambda item: (-len(item.request.questions), item.id))
        for name, question in record.request.questions.items():
            raw = record.model_dump()
            raw["request"]["questions"] = {name: question.model_dump()}
            raw["targets"] = {name: record.targets[name]}
            raw["tags"] += ["training_replay"]
            families[record.family].append(TrainingRecord.model_validate(raw))
    return dict(families)


def sample_replay(pool, count: int, rng: random.Random) -> list[TrainingRecord]:
    if count < 0 or (count and not pool):
        raise ValueError("replay needs a nonempty pool and a nonnegative count")
    names = sorted(pool)
    queues, output = {}, []
    for index in range(count):
        family = names[index % len(names)]
        if not queues.get(family):
            queues[family] = rng.sample(pool[family], len(pool[family]))
        output.append(queues[family].pop())
    rng.shuffle(output)
    return output
