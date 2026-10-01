"""Select text observations before model use; needs pyarrow only in this data tool."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def key(value):
    return hashlib.sha256(("veyra-foundation-v11-83:" + value).encode()).hexdigest()


def normalized(value):
    return " ".join(value.lower().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/foundation-v11-source"))
    args = parser.parse_args()
    output = args.source / "selected-text.json"
    if output.exists():
        raise FileExistsError("text selections are immutable")
    selected = []
    used_premises = set()
    # Protect official test and validation groups before selecting training observations.
    for upstream, quotas in (
        ("test", {"test": 100}),
        ("validation", {"dev": 40, "calibration": 40}),
        ("train", {"train": 200}),
    ):
        rows = pq.read_table(args.source / f"snli-{upstream}.parquet").to_pylist()
        candidates = defaultdict(list)
        for row in rows:
            if row["label"] not in (0, 1, 2):
                continue
            premise = normalized(row["premise"])
            if premise not in used_premises:
                candidates[row["label"]].append(row)
        for label, pool in sorted(candidates.items()):
            pool.sort(key=lambda r: key(r["premise"] + "\n" + r["hypothesis"]))
            cursor = 0
            for split, quota in quotas.items():
                count = 0
                while count < quota:
                    row = pool[cursor]
                    cursor += 1
                    premise = normalized(row["premise"])
                    if premise in used_premises:
                        continue
                    used_premises.add(premise)
                    selected.append(
                        {
                            "source": "snli",
                            "upstream_split": upstream,
                            "split": split,
                            "id": key(premise)[:24],
                            "label": label,
                            "text": "Premise: "
                            + row["premise"]
                            + "\nHypothesis: "
                            + row["hypothesis"],
                        }
                    )
                    count += 1
        # Exclude every official holdout premise, including unselected rows, from train.
        if upstream != "train":
            used_premises.update(normalized(row["premise"]) for row in rows)

    categories = json.loads((args.source / "banking-categories.json").read_text())
    order = sorted(categories, key=key)
    category_groups = {"train": order[:60], "dev_novel": order[60:68], "test_novel": order[68:]}
    pools = {}
    for upstream in ("train", "test"):
        with (args.source / f"banking-{upstream}.csv").open(encoding="utf-8", newline="") as f:
            pools[upstream] = list(csv.DictReader(f))
    test_texts = {normalized(row["text"]) for row in pools["test"]}
    used_texts = set()
    for category in categories:
        train_pool = sorted(
            [
                r
                for r in pools["train"]
                if r["category"] == category and normalized(r["text"]) not in test_texts
            ],
            key=lambda r: key(r["text"]),
        )
        quotas = (
            {"train": 12, "dev": 2, "calibration": 2}
            if category in order[:60]
            else ({"dev": 8, "calibration": 2} if category in order[60:68] else {})
        )
        cursor = 0
        for split, count in quotas.items():
            for _ in range(count):
                while normalized(train_pool[cursor]["text"]) in used_texts:
                    cursor += 1
                row = train_pool[cursor]
                cursor += 1
                used_texts.add(normalized(row["text"]))
                selected.append(
                    {
                        "source": "banking",
                        "upstream_split": "train",
                        "split": split,
                        "id": key(normalized(row["text"]))[:24],
                        "label": category,
                        "text": row["text"],
                        "novel_label": category not in order[:60],
                    }
                )
        test_pool = sorted(
            [r for r in pools["test"] if r["category"] == category], key=lambda r: key(r["text"])
        )
        count = 0
        for row in test_pool:
            if normalized(row["text"]) in used_texts:
                continue
            used_texts.add(normalized(row["text"]))
            selected.append(
                {
                    "source": "banking",
                    "upstream_split": "test",
                    "split": "test",
                    "id": key(normalized(row["text"]))[:24],
                    "label": category,
                    "text": row["text"],
                    "novel_label": category in order[68:],
                }
            )
            count += 1
            if count == 4:
                break
    document = {
        "seed": 83,
        "rows": selected,
        "banking_categories": category_groups,
        "banking_scope": "8-way candidate subsets, not the published full 77-class benchmark",
        "snli_group_scope": "Normalized premise disjoint; original image identities unavailable",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {"rows": len(selected), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
        )
    )


if __name__ == "__main__":
    main()
