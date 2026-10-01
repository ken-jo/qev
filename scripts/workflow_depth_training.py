"""Training-only memory policy and parameter fingerprints for the depth comparison."""

import hashlib

from workflow_depth_execution import configure_checkpointing


def install_memory_policy(model, threshold):
    """Use measured partial checkpointing, with full checkpointing on longer sequences."""
    counts = {"earlier12": 0, "all24": 0, "maximum_tokens": 0}

    def before_forward(module, args, kwargs):
        if not module.language_model.training:
            return
        identifiers = kwargs["input_ids"]
        if identifiers.shape[0] != 1:
            raise ValueError("the depth study requires one independent input per forward")
        tokens = int(identifiers.shape[1])
        strategy = "earlier12" if tokens <= threshold else "all24"
        configure_checkpointing(model, strategy)
        counts[strategy] += 1
        counts["maximum_tokens"] = max(counts["maximum_tokens"], tokens)

    handle = model.encoder.model.register_forward_pre_hook(before_forward, with_kwargs=True)
    return handle, counts


def group_hash(group):
    value = hashlib.sha256()
    for parameter in group["params"]:
        value.update(parameter.detach().cpu().numpy().tobytes())
    return value.hexdigest()
