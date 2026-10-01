"""Run pinned Laya checkpoints with identical text states and dynamic question criteria."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from train_foundation_head import write

from veyra.data import Source, TrainingRecord, read_records
from veyra.decision_metrics import observation, report
from veyra.schema import DecisionRequest


def official_records(path):
    source = Source(
        id="LocalLLaMA/typed-decisions",
        revision="d51d993547ad8355b1c25157fbc1fea0649e8ffa",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/LocalLLaMA/typed-decisions",
        upstream_split="test",
    )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        questions, gold = json.loads(case["questions"]), json.loads(case["gold"])
        for name, question in questions.items():
            target = {str(k): float(v) for k, v in gold[name]["probabilities"].items()}
            total = sum(target.values())
            rows.append(
                TrainingRecord(
                    id=case["id"] + ":" + name,
                    group_id=case["id"],
                    split="test",
                    family=case["workflow"],
                    language="en",
                    source=source,
                    request=DecisionRequest.model_validate(
                        {"state": {"text": case["state"]}, "questions": {name: question}}
                    ),
                    targets={name: {k: v / total for k, v in target.items()}},
                    tags=[
                        "domain:" + case["workflow"],
                        "gold_key:" + str(gold[name]["label"]),
                        "legacy_regression",
                        "teacher_distribution",
                    ],
                )
            )
    return rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=int, choices=[0, 1], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--max-len", type=int)
    args = parser.parse_args()
    if bool(args.records) == bool(args.official):
        raise ValueError("supply either a foundation corpus or official typed-decisions JSONL")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("evaluation output must be empty")
    references = json.loads(Path("runs/foundation-v11-laya/sources.json").read_text())
    reference = references[args.reference]
    folder = Path(reference["path"])
    if digest(folder / "model.safetensors") != reference["sha256"]["model.safetensors"]:
        raise ValueError("reference weights changed")
    records = (
        official_records(args.official)
        if args.official
        else [
            r
            for r in read_records(args.records)
            if r.split == args.split and not r.request.state.images
        ]
    )
    original_questions = {}
    if args.official:
        for line in args.official.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            for name, question in json.loads(case["questions"]).items():
                original_questions[case["id"] + ":" + name] = {name: question}
    protocol = {
        "arguments": vars(args),
        "reference": reference,
        "dataset_sha256": digest(args.records or args.official),
        "script_sha256": digest(Path(__file__)),
        "input": (
            "Same original text string, question instructions and criteria, one question per call"
        ),
        "runtime": (
            "laya==0.3.20, released temperatures; context default unless --max-len is supplied; "
            "no weight or temperature tuning"
        ),
        "probability_rounding": (
            "Published SDK rounds to four decimals; normalize returned distributions"
        ),
        "latency_scope": "Resident diagnostic, no HTTP or concurrent GPU load intended",
    }
    write(args.output / "protocol.json", protocol)
    sys.path.insert(0, str(Path(".cache/laya-sdk").resolve()))
    import laya

    torch.set_num_threads(4)
    agent = laya.load(str(folder), device="cuda")
    rows, times = [], []
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for index, record in enumerate(records, start=1):
            questions = {
                name: question.model_dump() for name, question in record.request.questions.items()
            }
            if args.official:
                questions = original_questions[record.id]
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = agent.predict(record.request.state.text, questions, max_len=args.max_len)
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - start))
            if agent.device.type != "cuda":
                raise RuntimeError("reference changed to CPU; rerun after resolving GPU capacity")
            for name, answer in result["answers"].items():
                probabilities = answer.get("probabilities")
                if record.request.questions[name].type == "noul":
                    probabilities = {"false": 1 - answer["noul"], "true": answer["noul"]}
                row = observation(record, name, probabilities, prediction=answer.get("choice"))
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 200 == 0:
                print(
                    json.dumps(
                        {"evaluated": index, "total": len(records), "reference": reference["repo"]}
                    ),
                    flush=True,
                )
    import numpy as np

    result = {
        "protocol": protocol,
        **report(rows, bootstrap=args.split == "test"),
        "device": torch.cuda.get_device_name(),
        "diagnostic_latency_ms": {
            "p50": float(np.percentile(times[3:], 50)),
            "p95": float(np.percentile(times[3:], 95)),
        },
        "predictions_sha256": digest(args.output / "predictions.jsonl"),
    }
    write(args.output / "evaluation.json", result)
    print(json.dumps({"overall": result["overall"], "by_domain": result["by_domain"]}), flush=True)


if __name__ == "__main__":
    main()
