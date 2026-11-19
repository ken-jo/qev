"""Evaluation metrics that keep hard gold labels distinct from soft targets."""

from collections import defaultdict

import numpy as np


def observation(record, name, probabilities, prediction=None, abstained=False):
    question = record.request.questions[name]
    keys = list(probabilities)
    p = np.array([probabilities[k] for k in keys], dtype=float)
    p /= p.sum()
    target = np.array([record.targets[name][k] for k in keys], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-6:
        raise ValueError("invalid probabilities")
    gold_tags = [tag[9:] for tag in record.tags if tag.startswith("gold_key:")]
    gold = gold_tags[0] if gold_tags else None
    if gold is None and target.max() > 1 - 1e-6:
        gold = keys[int(target.argmax())]
    winner = prediction or keys[int(p.argmax())]
    if winner not in keys or (gold is not None and gold not in keys):
        raise ValueError("invalid decision label")
    domains = [tag[7:] for tag in record.tags if tag.startswith("domain:")]
    views = [tag[5:] for tag in record.tags if tag.startswith("view:")]
    return {
        "id": record.id,
        "group": record.group_id,
        "question": name,
        "domain": domains[0] if domains else record.family,
        "family": record.family,
        "view": views[0] if views else "legacy",
        "modality": "image" if record.request.state.images else "text",
        "scope": "legacy_regression" if "legacy_regression" in record.tags else "fresh",
        "intent_novelty": "split_novel" if "novel_intent" in record.tags else "other",
        "uncertainty_target": (
            "known_conditional"
            if "known_conditional_distribution" in record.tags
            else "legacy_annotation"
        ),
        "type": question.type,
        "prediction": winner,
        "gold": gold,
        "correct": winner == gold if gold is not None else None,
        "probabilities": dict(zip(keys, p.tolist())),
        "targets": dict(zip(keys, target.tolist())),
        "confidence": float(p.max()),
        "nll": float(-(target * np.log(p.clip(1e-12))).sum()),
        "brier": float(np.square(p - target).sum()),
        "score_mae": float(abs(sum(float(k) * (x - y) for k, x, y in zip(keys, p, target))))
        if question.type == "score"
        else None,
        "abstained": bool(abstained),
    }


def summarize(rows, bootstrap=False):
    hard = [r for r in rows if r["correct"] is not None]
    result = {
        "questions": len(rows),
        "groups": len({r["group"] for r in rows}),
        "labeled_questions": len(hard),
        "correct": sum(r["correct"] for r in hard),
        "accuracy": float(np.mean([r["correct"] for r in hard])) if hard else None,
        "nll": float(np.mean([r["nll"] for r in rows])),
        "brier_vs_soft": float(np.mean([r["brier"] for r in rows])),
        "coverage": float(np.mean([not r["abstained"] for r in rows])),
    }
    accepted = [r for r in hard if not r["abstained"]]
    result["accepted_accuracy"] = (
        float(np.mean([r["correct"] for r in accepted])) if accepted else None
    )
    scores = [r["score_mae"] for r in rows if r["score_mae"] is not None]
    result["score_mae"] = float(np.mean(scores)) if scores else None
    result["score_mae_definition"] = "Absolute difference of ordinal expectations in index units"
    ece = 0.0
    for index in range(15):
        selected = [
            r
            for r in hard
            if index / 15 <= r["confidence"] < (index + 1) / 15
            or (index == 14 and r["confidence"] == 1)
        ]
        if selected:
            ece += (
                len(selected)
                / len(hard)
                * abs(
                    np.mean([r["confidence"] for r in selected])
                    - np.mean([r["correct"] for r in selected])
                )
            )
    result["ece_15_bins"] = float(ece) if hard else None
    if bootstrap and hard:
        grouped = defaultdict(list)
        for row in hard:
            grouped[row["group"]].append(float(row["correct"]))
        totals = np.array([[sum(v), len(v)] for v in grouped.values()])
        rng = np.random.default_rng(20260930)
        samples = totals[rng.integers(len(totals), size=(2000, len(totals)))].sum(1)
        result["group_bootstrap_accuracy_95_ci"] = np.quantile(
            samples[:, 0] / samples[:, 1], [0.025, 0.975]
        ).tolist()
    return result


def report(rows, bootstrap=False):
    result = {"overall": summarize(rows, bootstrap)}
    for field in ("domain", "modality", "view", "type", "scope"):
        result["by_" + field] = {
            value: summarize([r for r in rows if r[field] == value], bootstrap)
            for value in sorted({r[field] for r in rows})
        }
    result["by_domain_view"] = {
        domain + "/" + view: summarize(
            [r for r in rows if r["domain"] == domain and r["view"] == view], bootstrap
        )
        for domain, view in sorted({(r["domain"], r["view"]) for r in rows})
    }
    for domain, field in (("text_intent", "intent_novelty"), ("uncertainty", "uncertainty_target")):
        subset = [r for r in rows if r["domain"] == domain]
        if subset:
            result["by_" + field] = {
                value: summarize([r for r in subset if r[field] == value], bootstrap)
                for value in sorted({r[field] for r in subset})
            }
    return result
