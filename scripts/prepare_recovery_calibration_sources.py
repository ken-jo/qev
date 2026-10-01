"""Select unused public observations without loading any Veyra model or predictions."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return " ".join(text.lower().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config
    config = read(config_path)

    def stable(text):
        return hashlib.sha256((config["identity_namespace"] + ":" + text).encode()).hexdigest()

    source = Path("data/foundation-v11-source")
    root = Path(config["source_output"])
    output = root / "retention-selections.json"
    if output.exists():
        raise FileExistsError("fresh source selection is immutable")
    old_paths = [
        source / "selected-text.json",
        Path("data/workflow-v12-source/retention-selections.json"),
    ]
    old_paths.extend(Path(p) for p in config["exclude_retention_selections"])
    previous = read(old_paths[0])["rows"]
    for path in old_paths[1:]:
        previous.extend(read(path)["text_rows"])
    used_premises = {
        normalize(r["text"].split("\nHypothesis:")[0].removeprefix("Premise: "))
        for r in previous
        if r["source"] == "snli"
    }
    used_utterances = {normalize(r["text"]) for r in previous if r["source"] != "snli"}
    original_premises, original_utterances = set(used_premises), set(used_utterances)
    selections = []
    pools = defaultdict(list)
    for row in pq.read_table(source / "snli-validation.parquet").to_pylist():
        if row["label"] in (0, 1, 2) and normalize(row["premise"]) not in used_premises:
            pools[row["label"]].append(row)
    for label in range(3):
        selected = 0
        for row in sorted(
            pools[label], key=lambda r: stable(r["premise"] + "\n" + r["hypothesis"])
        ):
            identity = normalize(row["premise"])
            if identity in used_premises:
                continue
            used_premises.add(identity)
            selections.append(
                {
                    "source": "snli",
                    "upstream_split": "validation",
                    "split": "test" if "final_families" in config else "calibration",
                    "id": stable(identity)[:24],
                    "label": label,
                    "text": "Premise: " + row["premise"] + "\nHypothesis: " + row["hypothesis"],
                }
            )
            selected += 1
            if selected == config["quotas"]["snli_per_class"]:
                break
        if selected != config["quotas"]["snli_per_class"]:
            raise ValueError("insufficient unused SNLI premises")
    categories = read(source / "banking-categories.json")
    pools = defaultdict(list)
    with (source / "banking-test.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if normalize(row["text"]) not in used_utterances:
                pools[row["category"]].append(row)
    for category in categories:
        selected = 0
        for row in sorted(pools[category], key=lambda r: stable(r["text"])):
            identity = normalize(row["text"])
            if identity in used_utterances:
                continue
            used_utterances.add(identity)
            selections.append(
                {
                    "source": "banking77",
                    "upstream_split": "test",
                    "split": "test" if "final_families" in config else "calibration",
                    "id": stable(identity)[:24],
                    "label": category,
                    "text": row["text"],
                }
            )
            selected += 1
            if selected == config["quotas"]["banking77_per_class"]:
                break
        if selected != config["quotas"]["banking77_per_class"]:
            raise ValueError("insufficient unused BANKING77 utterances: " + category)
    used_images = set()
    image_record_paths = [
        Path(config["old_records"]),
        Path("data/foundation-v11/records.jsonl"),
        *(Path(p) for p in config["exclude_records"]),
    ]
    for path in image_record_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if value.get("image_sha256"):
                used_images.add(value["image_sha256"])
    pools = defaultdict(dict)
    cifar_path = Path("data/workflow-v12-source/cifar-test.parquet")
    for row in pq.read_table(cifar_path).to_pylist():
        raw = row["img"]["bytes"]
        identity = hashlib.sha256(raw).hexdigest()
        if identity not in used_images:
            pools[row["label"]][identity] = raw
    images = root / "cifar-images"
    images.mkdir(parents=True, exist_ok=False)
    proposals = []
    for label in range(10):
        for identity in sorted(pools[label], key=stable)[:100]:
            path = images / (identity + ".png")
            path.write_bytes(pools[label][identity])
            proposals.append({"sha256": identity, "label": label, "path": str(path)})
    if len(proposals) != 1000 or len(selections) != 304:
        raise ValueError("unexpected fresh-source population")
    sources = [
        *old_paths,
        *image_record_paths,
        source / "snli-validation.parquet",
        source / "banking-test.csv",
        cifar_path,
    ]
    result = {
        "seed": config["seed"],
        "identity_namespace": config["identity_namespace"],
        "calibration_only": "final_families" not in config,
        "no_model_outputs_used": True,
        "text_rows": selections,
        "banking_categories": categories,
        "image_candidates": proposals,
        "config_sha256": digest(config_path),
        "source_sha256": digest(Path(__file__)),
        "source_files": {str(path): digest(path) for path in sources},
        "previous_normalized_premises_excluded": len(original_premises),
        "previous_normalized_utterances_excluded": len(original_utterances),
        "previous_image_byte_hashes_excluded": len(used_images),
        "pyarrow_version": pyarrow.__version__,
    }
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {"fresh_text_observations": len(selections), "fresh_image_candidates": len(proposals)}
        )
    )


if __name__ == "__main__":
    main()
