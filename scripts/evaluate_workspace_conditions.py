"""Measure internal branch supervision and final decisions on development data only."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.evidence_data import original_branch
from veyra.option_model import OptionModel


def development_conditions(records):
    """Keep scoring annotations outside requests and reject non-development input."""
    if not records or any(
        record.split != "dev" or "counterfactual_pair" not in record.tags for record in records
    ):
        raise ValueError("expected controlled development policy pairs")
    groups = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
    branches = {}
    for name, group in groups.items():
        if len(group) != 2 or any(len(record.request.questions) != 1 for record in group):
            raise ValueError("expected two one-question counterfactuals per group")
        branches[name] = original_branch(sorted(group, key=lambda record: record.id)[0])
    return branches


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("condition diagnostics are immutable")
    manifest = json.loads((args.checkpoint / "manifest.json").read_text())
    if not manifest.get("reasoning_slots", 0):
        raise ValueError("this diagnostic requires a workspace checkpoint")
    records = [record for record in read_records(args.records) if record.split == "dev"]
    branches = development_conditions(records)
    model = OptionModel.load(args.checkpoint, local_files_only=True)
    rows = []
    for record in records:
        # Branch targets are used only below for scoring; the model receives the original request.
        predictions, _, conditions = model.forward_with_condition(
            record.request, args.records.parent
        )
        name, question = next(iter(record.request.questions.items()))
        candidates = candidates_for(question)
        key = candidates[int(predictions[name].argmax())].key
        branch = branches[record.group_id]
        rows.append(
            {
                "id": record.id,
                "group": record.group_id,
                "family": record.family,
                "type": question.type,
                "decision_correct": record.targets[name][key] == 1,
                "condition_correct": int(conditions[name].argmax()) == branch,
                "condition_logits": conditions[name].cpu().tolist(),
                "branch_target": branch,
            }
        )
        if len(rows) % 200 == 0:
            print(
                f"evaluated {len(rows)}/{len(records)} development decisions and conditions",
                flush=True,
            )

    def summarize(selected):
        paired = defaultdict(list)
        for row in selected:
            paired[row["group"]].append(row)
        pairs = [group for group in paired.values() if len(group) == 2]
        return {
            "questions": len(selected),
            "decision_accuracy": sum(row["decision_correct"] for row in selected) / len(selected),
            "condition_accuracy": sum(row["condition_correct"] for row in selected) / len(selected),
            "condition_correct_decision_wrong": sum(
                row["condition_correct"] and not row["decision_correct"] for row in selected
            ),
            "condition_wrong_decision_correct": sum(
                not row["condition_correct"] and row["decision_correct"] for row in selected
            ),
            "paired_groups": len(pairs),
            "condition_both_correct_accuracy": (
                sum(all(row["condition_correct"] for row in group) for group in pairs) / len(pairs)
                if pairs
                else None
            ),
            "decision_both_correct_accuracy": (
                sum(all(row["decision_correct"] for row in group) for group in pairs) / len(pairs)
                if pairs
                else None
            ),
        }

    report = {
        "evaluation_role": "development_condition_diagnostic",
        "dataset_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "checkpoint_sha256": manifest["weights_sha256"],
        "checkpoint_manifest_sha256": hashlib.sha256(
            (args.checkpoint / "manifest.json").read_bytes()
        ).hexdigest(),
        "evaluated_subset_sha256": hashlib.sha256(
            "".join(record.model_dump_json() + "\n" for record in records).encode()
        ).hexdigest(),
        "overall": summarize(rows),
        "by_modality": {
            kind: summarize([row for row in rows if row["family"].startswith(kind + "_")])
            for kind in ("text", "image")
        },
        "by_family": {
            kind: summarize([row for row in rows if row["family"] == kind])
            for kind in sorted({row["family"] for row in rows})
        },
        "by_type": {
            kind: summarize([row for row in rows if row["type"] == kind])
            for kind in sorted({row["type"] for row in rows})
        },
        "by_family_and_branch": {
            f"{family}/branch-{branch}": summarize(
                [row for row in rows if row["family"] == family and row["branch_target"] == branch]
            )
            for family, branch in sorted({(row["family"], row["branch_target"]) for row in rows})
        },
        "predictions": rows,
        "limitation": (
            "Auxiliary accuracy does not prove the final readout causally uses that representation."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2)
    )


if __name__ == "__main__":
    main()
