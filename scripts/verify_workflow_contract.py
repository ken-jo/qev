"""Measure backbone calls for real text and photograph requests on the frozen model."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from train_foundation_head import write

from veyra.data import read_records
from veyra.option_model import OptionModel
from veyra.schema import DecisionRequest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/foundation-v11/records.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("contract evidence is immutable")
    records = sorted(
        (r for r in read_records(args.records) if r.split == "test"), key=lambda r: r.id
    )
    photo = next(r for r in records if "domain:image_waste" in r.tags)
    text = next(r for r in records if not r.request.state.images)
    questions = {}
    for row in records:
        if row.group_id == photo.group_id:
            for question in row.request.questions.values():
                questions.setdefault(question.type, question)
    if set(questions) != {"choice", "score", "noul"}:
        raise ValueError("photo fixture must cover all three decision types")
    requests = {
        "text": text.request,
        "photograph_three_questions": DecisionRequest(
            state=photo.request.state, questions=questions
        ),
    }
    torch.set_num_threads(4)
    model = OptionModel.load(args.checkpoint, local_files_only=True, merge=True)
    forwards, results = [], {}
    hook = model.encoder.model.register_forward_hook(lambda *_: forwards.append(1))
    try:
        for name, request in requests.items():
            start = len(forwards)
            response = model.predict(request, args.records.parent.resolve())
            count = len(forwards) - start
            valid = all(
                all(math.isfinite(p) and 0 <= p <= 1 for p in answer["probabilities"].values())
                and abs(sum(answer["probabilities"].values()) - 1) <= 1e-5
                for answer in response["answers"].values()
            )
            results[name] = {
                "backbone_forwards": count,
                "generated_answer_tokens": response["usage"]["output_tokens"],
                "valid_probabilities": valid,
                "questions": len(request.questions),
                "types": sorted({q.type for q in request.questions.values()}),
                "input_tokens": response["usage"]["input_tokens"],
            }
    finally:
        hook.remove()
    result = {
        "weights_sha256": digest(args.checkpoint / "head.safetensors"),
        "manifest_sha256": digest(args.checkpoint / "manifest.json"),
        "records_sha256": digest(args.records),
        "source_sha256": digest(Path(__file__)),
        "gpu": torch.cuda.get_device_name(),
        "record_ids": [text.id, photo.id],
        "results": results,
        "passed": all(
            r["backbone_forwards"] == 1
            and r["generated_answer_tokens"] == 0
            and r["valid_probabilities"]
            for r in results.values()
        ),
    }
    write(args.output, result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
