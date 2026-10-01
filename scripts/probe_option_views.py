"""Development-only matched comparison of candidate-position consensus."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from veyra.data import read_records
from veyra.option_model import OptionModel
from veyra.schema import DecisionRequest


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/policy-v3/records.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = OptionModel.load(args.checkpoint, local_files_only=True)
    records = [record for record in read_records(args.records) if record.split == "dev"]
    reports = {}
    for views in (1, 2, 3):
        model.decision_views = views
        for modality in ("text", "image"):
            request = next(r.request for r in records if r.family.startswith(modality + "_"))
            for _ in range(3):
                model.predict(request, args.records.parent)
        rows = []
        for index, record in enumerate(records):
            raw = record.request.model_dump_json()
            torch.cuda.synchronize()
            started = time.perf_counter()
            answer = model.predict(DecisionRequest.from_json(raw), args.records.parent)
            json.dumps(answer)
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - started)
            for name, value in answer["answers"].items():
                probabilities = value["probabilities"]
                winner = max(probabilities, key=probabilities.get)
                rows.append(
                    {
                        "id": record.id,
                        "family": record.family,
                        "type": record.request.questions[name].type,
                        "correct": record.targets[name][winner],
                        "milliseconds": elapsed,
                        "probabilities": probabilities,
                    }
                )
            if (index + 1) % 100 == 0:
                print(f"views={views}: {index + 1}/{len(records)} development requests", flush=True)

        def summarize(selected):
            return {
                "questions": len(selected),
                "accuracy": np.mean([row["correct"] for row in selected]),
                "p95_ms": np.percentile([row["milliseconds"] for row in selected], 95),
            }

        reports[str(views)] = {
            "by_modality": {
                kind: summarize([row for row in rows if row["family"].startswith(kind + "_")])
                for kind in ("text", "image")
            },
            "by_type": {
                kind: summarize([row for row in rows if row["type"] == kind])
                for kind in ("choice", "score", "noul")
            },
            "by_family": {
                kind: summarize([row for row in rows if row["family"] == kind])
                for kind in sorted({row["family"] for row in rows})
            },
            "predictions": rows,
        }
        print(
            json.dumps(
                {
                    "views": views,
                    "by_modality": reports[str(views)]["by_modality"],
                    "by_type": reports[str(views)]["by_type"],
                }
            ),
            flush=True,
        )
        args.output.write_text(
            json.dumps(
                {"split": "dev", "matched_weights": str(args.checkpoint), "views": reports},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
