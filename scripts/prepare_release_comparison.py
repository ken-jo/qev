"""Freeze shared evaluation inputs before any final QEV/LAYA inference."""

import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

SOURCES = {
    "ag_news": {
        "repo": "fancyzhx/ag_news",
        "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        "file": "data/test-00000-of-00001.parquet",
        "criteria": {
            "world": "world news and international politics",
            "sports": "sports",
            "business": "business and economy",
            "sci_tech": "science and technology",
        },
        "field": "article",
        "question": "topic",
        "instructions": "What is the topic of `article`?",
        "laya_training_status": "In training according to upstream bench_apps.py",
    },
    "emotion": {
        "repo": "dair-ai/emotion",
        "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        "file": "split/test-00000-of-00001.parquet",
        "criteria": {k: k for k in ["sadness", "joy", "love", "anger", "fear", "surprise"]},
        "field": "text",
        "question": "emotion",
        "instructions": "Which emotion is most strongly expressed in `text`?",
        "laya_training_status": "Held out according to upstream bench_apps.py",
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--dataset-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.official) != "4881baae1cfd752311a58064cb2095c0457ee2c0b8914d1cfe2590e19933e7b3":
        raise ValueError("Official benchmark differs from the pinned 400-case test")
    args.output.mkdir(parents=True)
    rows = []
    for line in args.official.read_text("utf-8").splitlines():
        case = json.loads(line)
        gold = json.loads(case["gold"])
        for name, question in json.loads(case["questions"]).items():
            rows.append(
                {
                    "id": case["id"] + ":" + name,
                    "group": case["id"],
                    "suite": "typed_decisions",
                    "domain": case["workflow"],
                    "text": case["state"],
                    "questions": {name: question},
                    "gold": str(gold[name]["label"]),
                    "targets": {str(k): float(v) for k, v in gold[name]["probabilities"].items()},
                }
            )
    if len(rows) != 2000:
        raise ValueError("Expected all 2000 official questions")
    source_metadata = {}
    test_texts = {}
    for suite, source in SOURCES.items():
        url = (
            f"https://huggingface.co/datasets/{source['repo']}/resolve/"
            f"{source['revision']}/{source['file']}"
        )
        path = args.output / (suite + ".parquet")
        path.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        samples = pq.read_table(path).to_pylist()[:400]
        if len(samples) != 400:
            raise ValueError("Expected 400 test examples per application suite")
        keys = list(source["criteria"])
        for index, sample in enumerate(samples):
            record_id = f"{suite}:test:{index}"
            # Give both SDKs the same serialized state and non-null descriptions.
            text = json.dumps({source["field"]: sample["text"]}, ensure_ascii=False)
            test_texts[record_id] = normalized(sample["text"])
            winner = keys[int(sample["label"])]
            rows.append(
                {
                    "id": record_id,
                    "group": record_id,
                    "suite": suite,
                    "domain": suite,
                    "text": text,
                    "questions": {
                        source["question"]: {
                            "type": "choice",
                            "instructions": source["instructions"],
                            "criteria": source["criteria"],
                        }
                    },
                    "gold": winner,
                    "targets": {key: float(key == winner) for key in keys},
                }
            )
        source_metadata[suite] = {
            **source,
            "download_sha256": sha(path),
            "selected": "first 400 test rows",
            "class_counts": dict(Counter(int(r["label"]) for r in samples)),
        }
    # Audit the actual released corpus snapshots, not only a list of intended sources.
    import zipfile

    manifest = json.loads((args.dataset_package / "dataset-manifest.json").read_text("utf-8"))
    source_counts, matches, checked = Counter(), [], 0
    for corpus in manifest["corpora"]:
        archive_path = args.dataset_package / corpus["archive"]
        if sha(archive_path) != corpus["archive_sha256"]:
            raise ValueError("Training snapshot hash mismatch")
        with zipfile.ZipFile(archive_path) as archive:
            raw = archive.read(corpus["corpus"] + "/records.jsonl")
        if hashlib.sha256(raw).hexdigest() != corpus["published_records_sha256"]:
            raise ValueError("Training records hash mismatch")
        for line in raw.decode("utf-8").splitlines():
            record = json.loads(line)
            checked += 1
            source_counts[record["source"]["id"]] += 1
            state = normalized(record["request"]["state"].get("text", ""))
            for record_id, text in test_texts.items():
                if text and text in state:
                    matches.append(
                        {
                            "benchmark_id": record_id,
                            "corpus": corpus["corpus"],
                            "record_id": record["id"],
                            "split": record["split"],
                        }
                    )
    if matches or any(SOURCES[s]["repo"] in source_counts for s in SOURCES):
        raise ValueError("Application tasks overlap released adaptation data: " + str(matches[:5]))
    frozen = args.output / "requests.jsonl"
    frozen.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    protocol = {
        "version": "qev-0.1.1-final-comparison",
        "official_dataset_revision": "d51d993547ad8355b1c25157fbc1fea0649e8ffa",
        "official_dataset_sha256": sha(args.official),
        "sources": source_metadata,
        "requests_sha256": sha(frozen),
        "questions": dict(Counter(r["suite"] for r in rows)),
        "script_sha256": sha(Path(__file__)),
        "adaptation_data_audit": {
            "corpora": len(manifest["corpora"]),
            "rows_including_repeated_stages": checked,
            "source_counts": dict(source_counts),
            "normalized_substring_matches": matches,
            "scope": "Released adaptation snapshots; unknown backbone pretraining overlap",
        },
        "selection": "All 2000 typed decisions; first 400 AG News and Emotion test rows",
        "upstream_alignment": (
            "Application sample count, row selection and instructions follow LAYA bench_apps.py. "
            "Emotion null descriptions become label names for both SDKs. State JSON is identical."
        ),
        "evaluation": {
            "models": ["qev", "laya-base", "laya-specialist"],
            "one_question_per_call": True,
            "laya_max_len": 1024,
            "qev_max_tokens": 2048,
            "warmups_per_suite": 3,
            "torch_threads": 4,
            "device": "Single RTX 4060 Ti 8 GB, models run sequentially",
            "no_parameter_or_temperature_or_prompt_selection": True,
            "headline_accuracy": "All questions including abstained answers, explicit hard gold",
            "probability_metrics": "Same frozen decision_metrics.py for all models",
            "zero_shot": "No task-specific QEV adaptation or examples; not pretraining-clean",
            "typed_decisions": "Both QEV and specialist adapted; reused regression benchmark",
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "questions": protocol["questions"],
                "audit_rows": checked,
                "requests_sha256": sha(frozen),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
