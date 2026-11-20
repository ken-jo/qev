import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import probe_transfer_cvbench as probe  # noqa: E402


def make_rows(task, count):
    return [
        {
            "task": task,
            "filename": f"img/2D/{task.lower()}/photo_{i}.png",
            "question": f"How many things {i}?",
            "choices": ["3", "2", "1"],
            "answer": "(B)",
        }
        for i in range(count)
    ]


def test_quota_pairs_parses_task_limits():
    assert probe.quota_pairs("Count=150,Relation=150") == {"Count": 150, "Relation": 150}


def test_select_rows_is_stratified_deterministic_and_hash_ordered():
    rows = make_rows("Count", 40) + make_rows("Relation", 40)
    quotas = {"Count": 5, "Relation": 3}
    first = probe.select_rows(rows, quotas)
    second = probe.select_rows(list(reversed(rows)), quotas)
    assert [r["filename"] for r in first] == [r["filename"] for r in second]
    counts = {}
    for row in first:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    assert counts == quotas
    hashes = [
        hashlib.sha256(r["filename"].encode()).hexdigest() for r in first if r["task"] == "Count"
    ]
    assert hashes == sorted(hashes)
    pool = [
        hashlib.sha256(r["filename"].encode()).hexdigest() for r in rows if r["task"] == "Count"
    ]
    assert set(hashes) == set(sorted(pool)[:5])


def test_row_to_item_maps_answers_options_and_relative_images():
    row = make_rows("Count", 1)[0]
    item = probe.row_to_item(row, "2d")
    assert item["gold"] == "B"
    assert item["options"] == ["3", "2", "1"]
    assert item["source"] == "cvbench2d"
    assert item["subtask"] == "2d-count"
    assert item["images"] == ["img/2d/count/photo_0.png"]
    assert item["id"] == "cvb-2d-count-photo_0"


def test_probe_help_runs_without_downloading():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "probe_transfer_cvbench.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "--quotas" in result.stdout
