"""Audit canonical requests and actual pixel bytes without evaluating model predictions."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.packing import pack_request

path = Path("data/policy-v3/records.jsonl")
records = read_records(path)
pixels, pixel_splits, tasks, groups, slices = {}, {}, {}, defaultdict(list), Counter()
for record in records:
    pixel_digest = None
    if record.request.state.images:
        asset = path.parent / record.request.state.images[0].path
        if asset not in pixels:
            with Image.open(asset) as source:
                rgb = source.convert("RGB")
                pixels[asset] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                rgb.close()
        pixel_digest = pixels[asset]
        assert pixel_splits.setdefault(pixel_digest, record.split) == record.split
    signature = hashlib.sha256(
        (pack_request(record.request).text + str(pixel_digest)).encode()
    ).hexdigest()
    assert tasks.setdefault(signature, record.split) == record.split
    name, question = next(iter(record.request.questions.items()))
    candidates = candidates_for(question)
    winner = max(candidates, key=lambda c: record.targets[name][c.key])
    groups[record.group_id].append((record.split, question.type, winner.description))
    slices[record.split, record.family, record.language, question.type] += 1
assert all(len(pair) == 2 and pair[0][2] != pair[1][2] for pair in groups.values())
report = {
    "records": len(records),
    "groups": len(groups),
    "unique_pixels": len(pixel_splits),
    "canonical_request_cross_split_duplicates": 0,
    "pixel_cross_split_duplicates": 0,
    "counterfactual_groups_with_changed_target": len(groups),
    "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "slices": [
        {"split": key[0], "family": key[1], "language": key[2], "type": key[3], "records": count}
        for key, count in sorted(slices.items())
    ],
}
Path("reports/policy-v3-audit.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({k: v for k, v in report.items() if k != "slices"}, indent=2))
