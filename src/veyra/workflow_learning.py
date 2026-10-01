"""Shared training/development metrics for the frozen workflow release protocol."""

import torch

from veyra.candidates import candidates_for
from veyra.option_model import option_prompt
from veyra.transfer_learning import transfer_metrics

SHARES = {
    "workflow_known": 0.30,
    "workflow_new": 0.25,
    "uncertainty": 0.20,
    "image_waste": 0.075,
    "image_leaf": 0.025,
    "text_nli": 0.06,
    "text_intent": 0.06,
    "retention": 0.03,
}


def annotate(rows, records):
    lookup = {r.id: r for r in records}
    for row in rows:
        record = lookup[row["id"]]
        question = next(iter(record.request.questions.values()))
        _, positions = option_prompt(record.request, question)
        mapping = {c.key: p for c, p in zip(candidates_for(question), positions, strict=True)}
        critical = next(
            (t.removeprefix("critical_key:") for t in record.tags if t.startswith("critical_key:")),
            "",
        )
        row["critical_position"] = mapping.get(critical, -1)
        row["condition"] = next((t[10:] for t in record.tags if t.startswith("condition:")), "none")
        row["retention"] = "retention_replay" in record.tags


@torch.inference_mode()
def summarize_logits(logits, tensors, rows, indices, temperatures):
    selected = [rows[i] for i in indices.tolist()]
    types = tensors["types"][indices]
    scaled = logits.cpu().float() / torch.tensor(temperatures)[types, None]
    valid = tensors["valid"][indices]
    targets = tensors["targets"][indices]
    p = scaled.masked_fill(~valid, -1e9).softmax(-1)
    winner = p.argmax(-1)
    confidence = p.max(-1).values
    expected_error = 1 - targets.gather(1, winner[:, None]).squeeze(1)
    critical = torch.tensor([r["critical_position"] for r in selected])
    critical_mass = targets.gather(1, critical.clamp_min(0)[:, None]).squeeze(1)
    cost = expected_error + 4 * critical_mass * (critical >= 0) * (winner != critical)

    def summarize(mask):
        result = transfer_metrics(
            scaled[mask], targets[mask], valid[mask], tensors["gold"][indices][mask], types[mask]
        )
        chosen = torch.where(mask)[0]
        order = chosen[torch.argsort(confidence[chosen], descending=True, stable=True)]
        result["matched_coverage"] = {
            str(coverage): {
                "accepted": max(1, int(len(order) * coverage)),
                "expected_error": float(
                    expected_error[order[: max(1, int(len(order) * coverage))]].mean()
                ),
                "expected_cost": float(cost[order[: max(1, int(len(order) * coverage))]].mean()),
            }
            for coverage in (0.5, 0.8, 0.9, 1.0)
        }
        return result

    result = {"overall": summarize(torch.ones(len(selected), dtype=torch.bool))}
    for field in ("domain", "family", "condition"):
        result["by_" + field] = {
            value: summarize(torch.tensor([r[field] == value for r in selected]))
            for value in sorted({r[field] for r in selected})
        }
    return result


@torch.inference_mode()
def evaluate(model, tensors, rows, indices, temperatures):
    model.eval()
    logits = model(tensors["states"][indices], tensors["condition_logits"][indices])
    return summarize_logits(logits, tensors, rows, indices, temperatures)


def selection_key(metrics, baseline):
    domains, before = metrics["by_domain"], baseline["by_domain"]
    retention = all(
        domains[d]["accuracy"] >= before[d]["accuracy"] - 0.01 - 1e-9
        for d in ("photo_guard", "text_nli", "text_intent")
    )
    uncertainty, old = domains["uncertainty"], before["uncertainty"]
    probability = (
        uncertainty["nll"] < old["nll"]
        and uncertainty["brier_vs_soft"] < old["brier_vs_soft"]
        and uncertainty["matched_coverage"]["0.8"]["expected_cost"]
        < old["matched_coverage"]["0.8"]["expected_cost"]
    )
    improves = domains["workflow_new"]["accuracy"] > before["workflow_new"]["accuracy"]
    macro = sum(domains[d]["accuracy"] for d in ("workflow_known", "workflow_new")) / 2
    # Development eligibility guides selection; final release gates are checked separately.
    return (
        retention and improves and probability,
        retention,
        macro,
        -uncertainty["nll"],
        -uncertainty["matched_coverage"]["0.8"]["expected_cost"],
    )
