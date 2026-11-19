"""Matched differentiable and score-function objectives for reported distributions."""

import torch


def proper_reward(logits, targets, valid, types):
    """Log + 0.5 spherical score - 0.25 normalized ordinal RPS."""
    masked = logits.masked_fill(~valid, -1e9)
    logp = masked.log_softmax(-1)
    p = logp.exp()
    log_score = (targets * logp).sum(-1)
    spherical = (targets * p).sum(-1) / p.norm(dim=-1).clamp_min(1e-12)
    ranked = ((p.cumsum(-1) - targets.cumsum(-1)).square() * valid).sum(-1)
    ranked = ranked / (valid.sum(-1) - 1).clamp_min(1)
    return log_score + 0.5 * spherical - 0.25 * ranked * (types == 1)


def distribution_losses(logits, targets, valid, types, method="direct", sigma=0.15, samples=4):
    masked = logits.masked_fill(~valid, -1e9)
    ce = -(targets * masked.log_softmax(-1)).sum(-1)
    if method == "ce":
        return ce
    if method in {"direct", "cross"}:
        return -proper_reward(logits, targets, valid, types)
    if method not in {"direct_noise", "rloo"} or sigma <= 0 or samples < 2:
        raise ValueError("invalid exploration objective")
    count = valid.sum(-1, keepdim=True)
    location = (logits - (logits * valid).sum(-1, keepdim=True) / count) * valid
    noise = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * valid
    noise = (noise - noise.sum(-1, keepdim=True) / count) * valid
    if method == "direct_noise":
        reward = proper_reward(location.unsqueeze(0) + noise, targets, valid, types)
        return -reward.mean(0) + 0.5 * ce
    observations = location.detach().unsqueeze(0) + noise
    with torch.no_grad():
        reward = proper_reward(observations, targets, valid, types)
        baseline = (reward.sum(0, keepdim=True) - reward) / (samples - 1)
        advantage = reward - baseline
    log_density = -((observations - location.unsqueeze(0)).square() * valid).sum(-1) / (
        2 * sigma**2
    )
    return -(advantage * log_density).mean(0) + 0.5 * ce


def retention_losses(logits, parent_logits, valid):
    reference = (parent_logits / 2).masked_fill(~valid, -1e9).softmax(-1)
    student = (logits / 2).masked_fill(~valid, -1e9).log_softmax(-1)
    return 2 * (reference * (reference.clamp_min(1e-12).log() - student)).sum(-1)


def transport_losses(logits, valid, batch, rows):
    """Push a recognition distribution through a known training-only policy mapping."""
    positions = {
        global_index: local_index for local_index, global_index in enumerate(batch.tolist())
    }
    p = logits.masked_fill(~valid, -1e9).softmax(-1)
    logp = logits.masked_fill(~valid, -1e9).log_softmax(-1)
    result = torch.zeros(len(batch), device=logits.device)
    for local_index, global_index in enumerate(batch.tolist()):
        row = rows[global_index]
        if "transport_parent" not in row:
            continue
        if row["transport_parent"] not in positions:
            raise ValueError("transport source must be in the same grouped batch")
        source = p[positions[row["transport_parent"]]].detach()
        target = torch.zeros(16, device=logits.device)
        for origin, destination in enumerate(row["transport_map"]):
            if destination >= 0:
                target[destination] += source[origin]
        result[local_index] = (target * (target.clamp_min(1e-12).log() - logp[local_index])).sum()
    return result


def grouped_batches(indices, rows, batch_size, generator):
    groups = {}
    for index in indices.tolist():
        groups.setdefault(rows[index]["group"], []).append(index)
    units = list(groups.values())
    order = torch.randperm(len(units), generator=generator).tolist()
    pending = []
    for i in order:
        if pending and len(pending) + len(units[i]) > batch_size:
            yield torch.tensor(pending, dtype=torch.long)
            pending = []
        pending.extend(units[i])
    if pending:
        yield torch.tensor(pending, dtype=torch.long)


def foundation_selection(metrics, baseline):
    current, initial = metrics["by_domain"], baseline["by_domain"]
    domains = ("text_nli", "text_intent", "image_waste", "image_leaf")
    eligible = (
        all(current[d]["accuracy"] >= initial[d]["accuracy"] - 0.02 for d in domains)
        and current["retention"]["accuracy"] >= initial["retention"]["accuracy"] - 0.02
        and current["uncertainty"]["nll"] <= initial["uncertainty"]["nll"] + 0.02
    )
    accuracy = sum(current[d]["accuracy"] for d in domains) / len(domains)
    nll = sum(current[d]["nll"] for d in (*domains, "uncertainty")) / (len(domains) + 1)
    return bool(eligible), accuracy, -nll
