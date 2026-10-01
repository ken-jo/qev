"""Development-only decomposition into numeric comparison, condition, and outcome tasks.

The oracle below labels diagnostic questions only. Production inference never imports it.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from veyra.data import read_records
from veyra.evidence_data import policy_branches
from veyra.option_model import OptionModel
from veyra.schema import DecisionRequest


def truth(expression, values):
    if " OR " in expression:
        return any(truth(part, values) for part in expression.split(" OR "))
    if " AND " in expression:
        return all(truth(part, values) for part in expression.split(" AND "))
    expression = expression.strip("() ")
    interval = re.fullmatch(r"(\d+) <= (.+?) < (\d+)", expression)
    if interval:
        low, field, high = interval.groups()
        return int(low) <= values[field] < int(high)
    field, operator, threshold = re.fullmatch(r"(.+?) (>=|<) (\d+)", expression).groups()
    return values[field] >= int(threshold) if operator == ">=" else values[field] < int(threshold)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/policy-v3/records.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new diagnostic output")
    baseline = json.loads(args.baseline.read_text())
    if baseline["split"] != "dev" or baseline["device"] != args.device:
        raise ValueError("baseline must be development-only on the same device")
    if Path(baseline["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("baseline checkpoint mismatch")
    original = baseline["variants"]["trained_option_letters"]["predictions"]
    wanted = {row["id"] for row in original}
    groups = defaultdict(list)
    for record in read_records(args.records):
        if record.id in wanted:
            if record.split != "dev" or not record.family.startswith("text_"):
                raise ValueError("stage diagnosis is restricted to text development records")
            groups[record.group_id].append(record)
    torch.set_num_threads(args.threads)
    model = OptionModel.load(args.checkpoint, device=args.device, local_files_only=True)
    rows = []
    for group in groups.values():
        record = min(group, key=lambda item: item.id)
        values = {
            field.strip(): int(value)
            for field, value in re.findall(
                r"(?:Observation: |; )(.+?) = (\d+)", record.request.state.text
            )
        }
        predicates, _ = policy_branches(record)
        queries = []
        if len(predicates) == 1:
            condition = predicates[0]
            queries.append(
                (
                    "condition",
                    {
                        "type": "noul",
                        "instructions": (
                            f"Using the observation, is this condition true: {condition}?"
                        ),
                    },
                    "true" if truth(condition, values) else "false",
                )
            )
        else:
            winning = [i for i, condition in enumerate(predicates) if truth(condition, values)]
            if len(winning) != 1:
                raise ValueError("controlled range conditions must identify exactly one branch")
            queries.append(
                (
                    "condition",
                    {
                        "type": "choice",
                        "instructions": "Which condition is satisfied by the observation?",
                        "criteria": {str(i): condition for i, condition in enumerate(predicates)},
                    },
                    str(winning[0]),
                )
            )
        atoms = set()
        for predicate in predicates:
            for field in values:
                for operator, number in re.findall(re.escape(field) + r" (>=|<) (\d+)", predicate):
                    atoms.add(f"{field} {operator} {number}")
                for number in re.findall(r"(\d+) <= " + re.escape(field), predicate):
                    atoms.add(f"{field} >= {number}")
        for atom in sorted(atoms):
            queries.append(
                (
                    "atomic_comparison",
                    {
                        "type": "noul",
                        "instructions": f"Using the observation, is this condition true: {atom}?",
                    },
                    "true" if truth(atom, values) else "false",
                )
            )
        for stage, question, answer in queries:
            request = DecisionRequest.model_validate(
                {"state": record.request.state.model_dump(), "questions": {"condition": question}}
            )
            result = model.predict(request, args.records.parent)["answers"]["condition"]
            probabilities = result["probabilities"]
            rows.append(
                {
                    "group": record.group_id,
                    "family": record.family,
                    "stage": stage,
                    "question": question,
                    "expected": answer,
                    "correct": float(max(probabilities, key=probabilities.get) == answer),
                    "probabilities": probabilities,
                }
            )
        print(f"diagnosed {record.group_id}", flush=True)

    def summarize(selected):
        return {
            "questions": len(selected),
            "accuracy": sum(r["correct"] for r in selected) / len(selected),
        }

    report = {
        "split": "dev",
        "device": args.device,
        "checkpoint_sha256": hashlib.sha256(
            (args.checkpoint / "head.safetensors").read_bytes()
        ).hexdigest(),
        "baseline_report_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "original_decisions": summarize(original),
        "stages": {
            stage: summarize([r for r in rows if r["stage"] == stage])
            for stage in ("condition", "atomic_comparison")
        },
        "predictions": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
