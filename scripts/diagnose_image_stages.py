"""Actual-model visual attribute/condition diagnostics on dev or explicitly retired final data."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.evidence_data import policy_branches
from veyra.option_model import OptionModel
from veyra.policy_data import COLORS, SHAPES
from veyra.schema import DecisionRequest
from veyra.synthetic_audit import diagram_objects, visual_predicate


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--retirement", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic reports are immutable")
    records = [r for r in read_records(args.records) if r.split == args.split]
    subset_hash = hashlib.sha256(
        "".join(r.model_dump_json() + "\n" for r in records).encode()
    ).hexdigest()
    if args.split == "test":
        if not args.retirement:
            raise ValueError("final-set diagnosis requires an explicit retirement record")
        retirement = json.loads(args.retirement.read_text())
        if (
            retirement.get("new_role") != "regression"
            or retirement.get("subset_sha256") != subset_hash
        ):
            raise ValueError("this final subset has not been retired")
    weights_hash = hashlib.sha256((args.checkpoint / "head.safetensors").read_bytes()).hexdigest()
    baseline = json.loads(args.baseline.read_text())
    if baseline["checkpoint_sha256"] != weights_hash or baseline["split"] != args.split:
        raise ValueError("baseline checkpoint/split mismatch")
    predictions = {row["id"]: row for row in baseline["predictions"]}
    groups = defaultdict(list)
    for record in records:
        if record.family.startswith("image_"):
            groups[record.group_id].append(record)
    model = OptionModel.load(args.checkpoint, local_files_only=True)
    rows = []

    def score(state, question, target):
        request = DecisionRequest(state=state, questions={"probe": question})
        logits = model(request, args.records.parent)[0]["probe"]
        candidates = candidates_for(request.questions["probe"])
        key = candidates[int(logits.argmax())].key
        return {"correct": key == target, "picked": key, "target": target}

    for group in groups.values():
        anchor = sorted(group, key=lambda r: r.id)[0]
        objects = diagram_objects(args.records.parent / anchor.request.state.images[0].path)
        predicate = policy_branches(anchor)[0][0]
        truth = visual_predicate(objects, predicate)
        policy_correct = []
        for record in group:
            outcomes = policy_branches(record)[1]
            outcome = outcomes[0 if truth else 1]
            name, question = next(iter(record.request.questions.items()))
            candidates = candidates_for(question)
            if question.type == "noul":
                probe = question.instructions.rsplit("'", 2)[1]
                key = "true" if probe == outcome else "false"
            else:
                key = next(c.key for c in candidates if c.description == outcome)
            if record.targets[name][key] != 1:
                raise ValueError("pixel-based oracle disagrees with recorded policy target")
            row = predictions[record.id]
            picked = max(range(len(row["logits"])), key=row["logits"].__getitem__)
            policy_correct.append(row["targets"][picked] == 1)
        if anchor.family == "image_count_rule":
            choices = {str(i): f"Exactly {i} colored geometric objects." for i in range(5)}
            question = "How many colored geometric objects are visible in the image?"
            target = str(len(objects))
        else:
            side = "leftmost" if "leftmost" in predicate else "rightmost"
            color = anchor.family == "image_color_rule"
            choices = {name: name for name in (COLORS if color else SHAPES)}
            question = f"What is the {'color' if color else 'shape'} of the {side} object?"
            target = (objects[0] if side == "leftmost" else objects[-1])[0 if color else 1]
        attribute = score(
            anchor.request.state,
            {"type": "choice", "instructions": question, "criteria": choices},
            target,
        )
        condition = score(
            anchor.request.state,
            {
                "type": "noul",
                "instructions": f"Using the observation, is this condition true: {predicate}?",
            },
            "true" if truth else "false",
        )
        rows.append(
            {
                "group": anchor.group_id,
                "family": anchor.family,
                "objects": objects,
                "predicate": predicate,
                "policy_correct": policy_correct,
                "attribute": attribute,
                "condition": condition,
            }
        )
        if len(rows) % 32 == 0:
            print(f"diagnosed {len(rows)}/{len(groups)} image observation groups", flush=True)
    result = {
        "evaluation_role": "retired_final_diagnostic" if args.split == "test" else "dev_diagnostic",
        "checkpoint_sha256": weights_hash,
        "evaluated_subset_sha256": subset_hash,
        "pixel_policy_label_audit": "all selected image targets verified",
        "by_family": {},
        "predictions": rows,
    }
    for family in sorted({r["family"] for r in rows}):
        selected = [r for r in rows if r["family"] == family]
        policy = [correct for r in selected for correct in r["policy_correct"]]
        result["by_family"][family] = {
            "observation_groups": len(selected),
            "policy_accuracy": sum(policy) / len(policy),
            **{
                stage + "_accuracy": sum(r[stage]["correct"] for r in selected) / len(selected)
                for stage in ("attribute", "condition")
            },
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["by_family"], indent=2))


if __name__ == "__main__":
    main()
