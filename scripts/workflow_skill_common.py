"""Canonical supervision and fixed prerequisite diagnostics; no inference-time rule parsing."""

from collections import defaultdict

import torch
from train_workflow_backbone import forward

from veyra.candidates import candidates_for
from veyra.constants import QUESTION_TYPES
from veyra.option_model import option_prompt


def canonical_target(record):
    name, question = next(iter(record.request.questions.items()))
    if len(record.request.questions) != 1:
        raise ValueError("one skill question per request required")
    _, positions = option_prompt(record.request, question)
    target, valid = torch.zeros(16), torch.zeros(16, dtype=torch.bool)
    for candidate, position in zip(candidates_for(question), positions, strict=True):
        target[position] = record.targets[name][candidate.key]
        valid[position] = True
    return target, valid, QUESTION_TYPES.index(question.type)


@torch.inference_mode()
def assess_skills(model, records, root):
    model.eval()
    rows, saved_logits = [], []
    for record in records:
        logits, _ = forward(model, record, root)
        logits = logits[0].cpu()
        target, valid, type_id = canonical_target(record)
        logp = (
            (logits / model.calibration.temperatures[type_id])
            .masked_fill(~valid, -1e9)
            .log_softmax(-1)
        )
        probability = logp.exp()
        winner = int(probability.argmax())
        hard = bool(target.max() == 1)
        kind = next(t[7:] for t in record.tags if t.startswith("domain:"))
        rows.append(
            {
                "id": record.id,
                "kind": kind,
                "type": QUESTION_TYPES[type_id],
                "family": record.family,
                "hard_correct": float(target[winner]) if hard else None,
                "expected_error": float(1 - target[winner]),
                "nll": float(-(target * logp).sum()),
                "brier": float((probability - target).square().sum()),
            }
        )
        saved_logits.append(logits)
    groups = defaultdict(list)
    for row in rows:
        for key in ("overall", row["kind"], row["kind"] + "/" + row["type"]):
            groups[key].append(row)
    result = {}
    for key, values in groups.items():
        hard = [v["hard_correct"] for v in values if v["hard_correct"] is not None]
        result[key] = {
            "questions": len(values),
            "hard_questions": len(hard),
            "hard_accuracy": sum(hard) / len(hard) if hard else None,
        }
        result[key].update(
            {
                metric: sum(v[metric] for v in values) / len(values)
                for metric in ("expected_error", "nll", "brier")
            }
        )
    return result, torch.stack(saved_logits)
