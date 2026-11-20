"""Paired out-of-distribution transfer probe on CV-Bench.

Evaluates the base checkpoint and one or more candidate deltas on a deterministic
stratified sample of CV-Bench (nyu-visionx/CV-Bench, Apache-2.0, Cambrian-1):
real photographs with multiple-choice counting, relation, depth and distance
questions. No CV-Bench row was ever used for training or checkpoint selection, so
a delta that only memorized the synthetic distribution shows up here. Models are
compared in one process on the identical sample; the item conversion mirrors the
mix-training item schema and is cached under --work-dir. Read paired deltas from one
run rather than absolute scores across runs: near-tie questions flip between MPS runs
and per-task absolutes move by several points.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from option_mix_common import load_items, load_student
from train_option_mix import evaluate

REPO_ID = "nyu-visionx/CV-Bench"
FILES = {
    "2d": ("test_2d.jsonl", "test_2d.parquet"),
    "3d": ("test_3d.jsonl", "test_3d.parquet"),
}
DEFAULT_QUOTAS = "Count=150,Relation=150,Depth=100,Distance=100"


def quota_pairs(value):
    pairs = {}
    for part in value.split(","):
        task, _, limit = part.partition("=")
        pairs[task.strip()] = int(limit)
    return pairs


def select_rows(rows, quotas):
    """Deterministic stratified sample: per task, the lowest sha256(filename)."""
    chosen = []
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    for task, limit in quotas.items():
        pool = sorted(
            by_task[task], key=lambda r: hashlib.sha256(r["filename"].encode()).hexdigest()
        )
        chosen.extend(pool[:limit])
    return chosen


def row_to_item(row, kind):
    return {
        "id": f"cvb-{kind}-{row['task'].lower()}-{Path(row['filename']).stem}",
        "source": f"cvbench{kind}",
        "subtask": f"{kind}-{row['task'].lower()}",
        "images": [f"img/{kind}/{Path(row['filename']).name}"],
        "question": row["question"],
        "options": row["choices"],
        "gold": row["answer"].strip("()"),
    }


def download(kind, name):
    return Path(hf_hub_download(repo_id=REPO_ID, filename=name, repo_type="dataset"))


def build_sample(work_dir, quotas):
    """Materialize the sampled items (and images) under work_dir once, then reuse."""
    stamp = {"quotas": quotas, "files": {k: v for k, v in FILES.items()}}
    stamp_path = work_dir / "sample.json"
    items_path = work_dir / "items.jsonl"
    if stamp_path.exists() and json.loads(stamp_path.read_text()) == stamp and items_path.exists():
        return items_path
    work_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for kind, (jsonl_name, parquet_name) in FILES.items():
        rows = _rows(download(kind, jsonl_name))
        present = {task for task in quotas if any(r["task"] == task for r in rows)}
        chosen = select_rows(rows, {task: quotas[task] for task in present})
        images = {}
        for row in _parquet_rows(download(kind, parquet_name)):
            blob = row["image"]
            images[row["filename"]] = blob["bytes"] if isinstance(blob, dict) else blob
        for row in chosen:
            rel = f"img/{kind}/{Path(row['filename']).name}"
            dest = work_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(images[row["filename"]])
            items.append(row_to_item(row, kind))
        print(f"cvbench {kind}: {len(chosen)} items", file=sys.stderr, flush=True)
    with open(items_path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    stamp_path.write_text(json.dumps(stamp))
    return items_path


def _rows(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _parquet_rows(path):
    import pyarrow.parquet as parquet

    return parquet.ParquetFile(path).read().to_pylist()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="released checkpoint directory")
    parser.add_argument(
        "--delta",
        action="append",
        default=[],
        type=Path,
        help="candidate safetensors delta (repeatable)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--work-dir", type=Path, default=".cache/cvbench")
    parser.add_argument(
        "--quotas", default=DEFAULT_QUOTAS, help="Task=N list, e.g. 'Count=150,Depth=100'"
    )
    parser.add_argument(
        "--pixels", type=int, default=262144, help="per-image pixel budget (matches mix training)"
    )
    parser.add_argument("--total-pixels", type=int, default=524288)
    parser.add_argument("--output", type=Path, help="write the JSON report here as well")
    args = parser.parse_args()

    quotas = quota_pairs(args.quotas)
    items_path = build_sample(args.work_dir, quotas)
    examples, _ = load_items([items_path], 0)
    groups = {}
    for example in examples:
        groups.setdefault(example["item"]["subtask"], []).append(example)

    candidates = [("", None)] + [(delta.stem, delta) for delta in args.delta]
    report = []
    for tag, delta in candidates:
        model = load_student(
            args.base, args.device, args.cache_dir, args.max_tokens, initialize=delta
        )
        per_task = {}
        total_n = total_right = 0
        for name, group in sorted(groups.items()):
            result = evaluate(model, group, [args.pixels], args.total_pixels)
            for kind, scores in result.items():
                if kind == "select":
                    continue
                per_task[name] = scores
                total_n += scores["n"]
                total_right += (
                    scores["right"] if "right" in scores else scores["accuracy"] * scores["n"]
                )
        row = {
            "model": tag or "base",
            "delta": str(delta) if delta else None,
            "overall": {"accuracy": total_right / total_n, "n": total_n},
            "per_task": per_task,
        }
        report.append(row)
        print(json.dumps(row), flush=True)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
