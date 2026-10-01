"""Summarize development-only errors without another GPU inference run."""

import argparse
import json
from collections import Counter
from pathlib import Path

from veyra.data import read_records

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--records", type=Path, default=Path("data/policy-v3/records.jsonl"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = json.loads(args.predictions.read_text())
if report["split"] != "dev":
    raise ValueError("This diagnostic is restricted to development data")
rows = report["views"]["1"]["predictions"]
records = {r.id: r for r in read_records(args.records) if r.split == "dev"}
counts, errors, high_confidence = Counter(), [], []
for row in rows:
    record = records[row["id"]]
    if row["correct"] == 1:
        continue
    counts[(row["family"], row["type"])] += 1
    item = {
        "id": record.id,
        "family": record.family,
        "type": row["type"],
        "state": record.request.state.model_dump(),
        "questions": {k: q.model_dump() for k, q in record.request.questions.items()},
        "targets": record.targets,
        "probabilities": row["probabilities"],
    }
    errors.append(item)
    if max(row["probabilities"].values()) >= 0.9:
        high_confidence.append(record.id)
result = {
    "split": "dev",
    "source_predictions": str(args.predictions),
    "errors": len(errors),
    "by_family_and_type": [
        {"family": family, "type": kind, "errors": count}
        for (family, kind), count in sorted(counts.items())
    ],
    "high_confidence_error_ids": high_confidence,
    "error_details": errors,
}
args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "error_details"}, indent=2))
