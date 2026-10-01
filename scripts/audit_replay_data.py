"""Check replay isolation without evaluating any model on held-out data."""

import hashlib
import json
from pathlib import Path

from veyra.data import read_records
from veyra.replay import audit_replay, replay_pool

primary_path = Path("data/policy-v6/records.jsonl")
replay_path = Path("data/starter-v2/records.jsonl")
primary = read_records(primary_path)
replay = [record for record in read_records(replay_path) if record.split == "train"]
report = {
    **audit_replay(primary, primary_path.parent, replay, replay_path.parent),
    "primary_sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
    "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
    "single_question_pool_by_family": {k: len(v) for k, v in replay_pool(replay).items()},
    "model_evaluation": False,
}
Path("reports/replay-data-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
