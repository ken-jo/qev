"""Check refreshed labels from numeric evidence and actual pixels; never run a model."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.evidence_data import policy_branches
from veyra.synthetic_audit import diagram_objects, numeric_predicate, visual_predicate

parser = argparse.ArgumentParser()
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--protocol", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("audit reports are immutable")
protocol = json.loads(args.protocol.read_text())
records = read_records(args.records)
if hashlib.sha256(args.records.read_bytes()).hexdigest() != protocol["records_sha256"]:
    raise ValueError("dataset changed after freezing")
counts, images = Counter(), {}
for record in records:
    predicates, outcomes = policy_branches(record)
    if record.family.startswith("text_"):
        values = {
            field.strip(): int(value)
            for field, value in re.findall(r"([^;:]+?) = (\d+)", record.request.state.text)
        }
        truths = [numeric_predicate(predicate, values) for predicate in predicates]
    else:
        path = args.records.parent / record.request.state.images[0].path
        if path not in images:
            images[path] = diagram_objects(path)
        truths = [visual_predicate(images[path], predicate) for predicate in predicates]
    branch = next((i for i, value in enumerate(truths) if value), len(predicates))
    answer = outcomes[branch]
    for name, question in record.request.questions.items():
        if question.type == "noul":
            probe = re.search(r"Is the policy outcome '([^']+)'\?", question.instructions).group(1)
            expected = {"false": float(answer != probe), "true": float(answer == probe)}
        else:
            expected = {c.key: float(c.description == answer) for c in candidates_for(question)}
        if record.targets[name] != expected:
            raise ValueError(f"independent label mismatch: {record.id}")
    counts[record.split] += 1
final_hash = hashlib.sha256(
    "".join(r.model_dump_json() + "\n" for r in records if r.split == "test").encode()
).hexdigest()
if final_hash != protocol["final_sha256"]:
    raise ValueError("frozen final changed")
report = {
    "passed": True,
    "model_evaluated": False,
    "records_verified": len(records),
    "by_split": counts,
    "images_decoded_and_verified": len(images),
    "dataset_sha256": protocol["records_sha256"],
    "final_sha256": final_hash,
    "method": "Independent numeric comparisons and pixel-derived colors/shapes/counts",
}
args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
