"""Training-only distribution agreement across equivalent typed questions."""

from collections import defaultdict

import torch

from veyra.candidates import candidates_for
from veyra.option_model import option_prompt


def aligned_training_groups(rows, records):
    lookup = {record.id: record for record in records if record.split == "train"}
    grouped = defaultdict(dict)
    for index, row in enumerate(rows):
        if row["split"] != "train" or row["domain"] != "workflow_new":
            continue
        record = lookup[row["id"]]
        if "view:complete" not in record.tags:
            continue
        name, question = next(iter(record.request.questions.items()))
        _, positions = option_prompt(record.request, question)
        candidates = candidates_for(question)
        if question.type in {"choice", "score"}:
            mapped = {
                candidate.description: position
                for candidate, position in zip(candidates, positions, strict=True)
            }
            labels = {
                candidate.description: record.targets[name][candidate.key]
                for candidate in candidates
            }
            meanings = ["Outcome PROCEED.", "Outcome REVIEW.", "Outcome HOLD."]
            if set(mapped) != set(meanings):
                raise ValueError("unsupported training outcome vocabulary")
            semantic = [mapped[key] for key in meanings]
            target = [labels[key] for key in meanings]
        else:
            mapped = {
                candidate.key: position
                for candidate, position in zip(candidates, positions, strict=True)
            }
            semantic = [mapped["false"], mapped["true"]]
            target = [record.targets[name]["false"], record.targets[name]["true"]]
        if question.type in grouped[record.group_id]:
            raise ValueError("duplicate complete typed view")
        grouped[record.group_id][question.type] = {
            "index": index,
            "positions": semantic,
            "target": target,
            "state": record.request.state.model_dump_json(),
        }
    links = []
    for group, views in grouped.items():
        if set(views) != {"choice", "score", "noul"}:
            raise ValueError("complete typed group is missing a view")
        choice, score, noul = (views[k] for k in ("choice", "score", "noul"))
        projection = [sum(choice["target"][:2]), choice["target"][2]]
        if (
            len({v["state"] for v in views.values()}) != 1
            or choice["target"] != score["target"]
            or any(abs(a - b) > 1e-6 for a, b in zip(projection, noul["target"], strict=True))
        ):
            raise ValueError("typed views do not describe the same state and outcome")
        links.append(
            {
                "group": group,
                "views": [
                    {key: value for key, value in view.items() if key in {"index", "positions"}}
                    for view in (choice, score, noul)
                ],
            }
        )
    return links


def js_divergence(distributions):
    stack = torch.stack(distributions)
    center = stack.mean(0)
    return (stack * (stack.clamp_min(1e-8).log() - center.clamp_min(1e-8).log())).sum(-1).mean()


def consistency_losses(logits, valid, batch, links_by_index):
    local = {index: position for position, index in enumerate(batch.tolist())}
    probabilities = logits.masked_fill(~valid, -1e9).softmax(-1)
    losses = torch.zeros(len(batch), device=logits.device)
    for index in local:
        if index not in links_by_index:
            continue
        link = links_by_index[index]
        views = link["views"]
        if any(view["index"] not in local for view in views):
            raise ValueError("typed consistency groups must stay in one training batch")
        choice, score, noul = [
            probabilities[local[view["index"]], view["positions"]] for view in views
        ]
        binary_choice = torch.stack((choice[:2].sum(), choice[2]))
        binary_score = torch.stack((score[:2].sum(), score[2]))
        agreement = js_divergence([choice, score]) + js_divergence(
            [
                binary_choice,
                binary_score,
                noul,
            ]
        )
        for view in views:
            losses[local[view["index"]]] = agreement
    return losses
