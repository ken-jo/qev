"""Explain decision errors and confidence ordering without fitting or selecting a policy."""

import math
from collections import Counter, defaultdict

import torch

from veyra.candidates import candidates_for
from veyra.constants import QUESTION_TYPES
from veyra.option_model import option_prompt


@torch.inference_mode()
def diagnose(logits, tensors, rows, indices, records, temperatures):
    types = tensors["types"][indices]
    valid = tensors["valid"][indices]
    targets, gold = tensors["targets"][indices], tensors["gold"][indices]
    probabilities = (
        (logits / torch.tensor(temperatures)[types, None]).masked_fill(~valid, -1e9).softmax(-1)
    )
    confidence, prediction = probabilities.max(-1)
    error = 1 - targets.gather(1, prediction[:, None]).squeeze(1)
    error = torch.where(gold >= 0, (prediction != gold).float(), error)
    selected_rows = [rows[index] for index in indices.tolist()]

    def summary(selected):
        count = len(selected)
        required = max(1, math.ceil(count * 0.6))
        actual = selected[torch.argsort(confidence[selected], descending=True, stable=True)][
            :required
        ]
        ideal = selected[torch.argsort(error[selected], stable=True)][:required]
        hard = selected[gold[selected] >= 0]
        return {
            "questions": count,
            "hard_label_accuracy": float((prediction[hard] == gold[hard]).float().mean())
            if len(hard)
            else None,
            "expected_error_full": float(error[selected].mean()),
            "mean_confidence": float(confidence[selected].mean()),
            "minimum_required_answers_at_60pct": required,
            "actual_fixed_count_error_at_60pct": float(error[actual].mean()),
            "ideal_ranking_of_same_predictions_error_at_60pct": float(error[ideal].mean()),
            "high_confidence_hard_errors_ge_0_8": int(
                ((confidence[hard] >= 0.8) & (prediction[hard] != gold[hard])).sum()
            ),
        }

    by_type = {
        kind: summary(torch.where(types == index)[0]) for index, kind in enumerate(QUESTION_TYPES)
    }
    strata = defaultdict(list)
    groups = defaultdict(dict)
    for index, row in enumerate(selected_rows):
        strata[row["domain"], row["type"]].append(index)
        if row["domain"] != "workflow_new":
            continue
        record = records[row["id"]]
        question = next(iter(record.request.questions.values()))
        _, positions = option_prompt(record.request, question)
        if question.type in {"choice", "score"}:
            candidates = candidates_for(question)
            matched = next(
                candidate.description
                for candidate, pos in zip(candidates, positions, strict=True)
                if pos == int(prediction[index])
            )
            groups[record.group_id][question.type] = (
                matched,
                bool(prediction[index] == gold[index]),
            )
    table, disagreements = Counter(), 0
    for views in groups.values():
        if set(views) != {"choice", "score"}:
            raise ValueError("incomplete complete-evidence typed pair")
        choice, score = views["choice"], views["score"]
        table[str((choice[1], score[1]))] += 1
        disagreements += choice[0] != score[0]
    return {
        "by_type": by_type,
        "by_domain_and_type": {
            "/".join(key): summary(torch.tensor(value)) for key, value in sorted(strata.items())
        },
        "equivalent_choice_score": {
            "groups": len(groups),
            "semantic_disagreements": disagreements,
            "correctness_counts_choice_then_score": dict(table),
        },
        "used_for_selection": False,
        "limitation": (
            "Ideal error ordering knows which existing predictions are wrong; it is not a model, "
            "deployed policy or accuracy improvement. Fixed-count prefixes can split ties. "
            "These development diagnostics add no candidate or selection criterion."
        ),
    }
