"""Diagnostic only: compare full and single-question requests on development records."""

import argparse
import json
from pathlib import Path

from veyra.data import read_records
from veyra.model import VeyraModel
from veyra.schema import DecisionRequest

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
args = parser.parse_args()
root = Path("data/starter-v1")
model = VeyraModel.load(args.checkpoint, local_files_only=True)
records = read_records(root / "records.jsonl")
results = []
for family in ("diagram_attributes", "text_policy", "leaf_photo", "missing_evidence"):
    selected = [r for r in records if r.split == "dev" and r.family == family][:8]
    for record in selected:
        full = model.predict(record.request, image_root=root)["answers"]
        for name, question in record.request.questions.items():
            request = DecisionRequest(state=record.request.state, questions={name: question})
            answer = model.predict(request, image_root=root)["answers"][name]
            target = record.targets[name]
            one = max(answer["probabilities"], key=answer["probabilities"].get)
            multi = max(full[name]["probabilities"], key=full[name]["probabilities"].get)
            results.append(
                {
                    "family": family,
                    "id": record.id,
                    "question": name,
                    "single_correct": target[one],
                    "multi_correct": target[multi],
                }
            )
for family in sorted({x["family"] for x in results}):
    rows = [x for x in results if x["family"] == family]
    print(
        json.dumps(
            {
                "family": family,
                "questions": len(rows),
                "single_accuracy": sum(x["single_correct"] for x in rows) / len(rows),
                "multi_accuracy": sum(x["multi_correct"] for x in rows) / len(rows),
            }
        ),
        flush=True,
    )
Path("reports/subset-diagnostic-v1.json").write_text(json.dumps(results, indent=2) + "\n")
