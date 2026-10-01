"""Proper distribution losses and development metrics for mixed-domain transfer."""

import torch
from torch import nn

from veyra.binding_head import ConditionedReadout


class TransferReadout(nn.Module):
    """The release's existing readouts; this introduces no inference-time module."""

    def __init__(self, parent_manifest, parent_state):
        super().__init__()
        binding = parent_manifest["binding_head"]
        self.readout = nn.Linear(2048, 16, bias=False)
        self.binding_head = ConditionedReadout(binding["rank"], binding["mode"])
        self.load_state_dict(
            {
                key: value
                for key, value in parent_state.items()
                if key == "readout.weight" or key.startswith("binding_head.")
            },
            strict=True,
        )

    def forward(self, states, condition_logits):
        return self.readout(states) + self.binding_head(states, condition_logits)


def transfer_losses(logits, targets, valid, types, parent_logits=None, retention=None):
    """Soft cross-entropy, ordinal CDF error, and optional retention distillation."""
    masked = logits.masked_fill(~valid, -1e9)
    log_probs = masked.log_softmax(-1)
    probabilities = masked.softmax(-1)
    ce = -(targets * log_probs).sum(-1)
    ranked = (probabilities.cumsum(-1) - targets.cumsum(-1)).square().sum(-1)
    ranked = ranked / (valid.sum(-1) - 1).clamp_min(1)
    losses = ce + 0.25 * ranked * (types == 1)
    if parent_logits is not None:
        if retention is None:
            raise ValueError("retention mask is required with parent logits")
        teacher = (parent_logits / 2).masked_fill(~valid, -1e9).softmax(-1)
        student_log = (logits / 2).masked_fill(~valid, -1e9).log_softmax(-1)
        kl = (teacher * (teacher.clamp_min(1e-12).log() - student_log)).sum(-1)
        losses = losses + 0.5 * 4 * kl * retention
    return losses


@torch.inference_mode()
def transfer_metrics(logits, targets, valid, gold, types):
    masked = logits.masked_fill(~valid, -1e9)
    probabilities = masked.softmax(-1)
    labeled = gold >= 0
    predictions = probabilities.argmax(-1)
    nll = -(targets * masked.log_softmax(-1)).sum(-1)
    levels = torch.arange(logits.shape[1], dtype=logits.dtype, device=logits.device)
    scores = types == 1
    return {
        "questions": int(len(logits)),
        "labeled_questions": int(labeled.sum()),
        "accuracy": float((predictions[labeled] == gold[labeled]).float().mean())
        if labeled.any()
        else None,
        "nll": float(nll.mean()),
        "brier_vs_soft": float((probabilities - targets).square().sum(-1).mean()),
        "score_mae": float(((probabilities - targets) * levels).sum(-1)[scores].abs().mean())
        if scores.any()
        else None,
        "mean_confidence": float(probabilities.max(-1).values.mean()),
    }


def selection_key(metrics, baseline):
    workflow = metrics["by_domain"]["workflow"]
    photo = metrics["by_domain"]["photo"]
    retention = metrics["by_domain"]["retention"]
    old = baseline["by_domain"]
    eligible = (
        workflow["accuracy"] >= old["workflow"]["accuracy"]
        and photo["accuracy"] >= old["photo"]["accuracy"]
        and retention["accuracy"] >= old["retention"]["accuracy"] - 0.02
    )
    score = 0.6 * workflow["accuracy"] + 0.4 * photo["accuracy"]
    return eligible, score, -0.5 * (workflow["nll"] + photo["nll"])
