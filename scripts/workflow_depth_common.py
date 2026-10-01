"""Extend existing adapters without changing deployment encoding or adding inference modules."""

import torch

from veyra.option_model import LoRALinear, install_adapters


def layer_number(name):
    return int(name.split("language_model.layers.", 1)[1].split(".", 1)[0])


def expand_depth(model):
    previous = tuple(model.adapter_names)
    total = len(model.encoder.model.language_model.layers)
    old_depth = model.adaptation["layers"]
    boundary = total - old_depth
    if total != 24 or old_depth != 12 or not previous:
        raise ValueError("this feasibility design requires the declared 12-of-24-layer parent")
    if {layer_number(name) for name in previous} != set(range(boundary, total)):
        raise ValueError("parent adapter placement changed")
    added = install_adapters(
        model.encoder.model, total, model.adaptation["rank"], model.adaptation["alpha"]
    )
    if set(added) & set(previous) or {layer_number(name) for name in added} != set(range(boundary)):
        raise ValueError("earlier adapter placement differs from the declared expansion")
    for name in added:
        layer = model.encoder.model.get_submodule(name)
        if not isinstance(layer, LoRALinear) or torch.count_nonzero(layer.lora_b):
            raise ValueError("new adapters must initially have exactly zero output")
    model.adapter_names = sorted((*previous, *added))
    model.adaptation = {**model.adaptation, "layers": total}
    model.condition_readout.requires_grad_(False)
    model.encoder.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if not all(layer.gradient_checkpointing for layer in model.encoder.model.language_model.layers):
        raise ValueError("language-layer gradient checkpointing was not enabled")
    return {"previous_layers": old_depth, "expanded_layers": total, "boundary": boundary}


def training_mode(model):
    language = model.encoder.model.language_model
    if language.config.attention_dropout != 0 or any(
        isinstance(module, torch.nn.Dropout) and module.p != 0 for module in language.modules()
    ):
        raise ValueError("matched checkpointed training requires zero language dropout")
    model.train()
    model.encoder.model.eval()
    # Checkpointing is activated by decoder-layer training mode. The vision stack stays in eval.
    language.train()


def optimizer_groups(model, earlier_lr, later_lr, readout_lr):
    earlier, later = [], []
    for name, parameter in model.named_parameters():
        if ".lora_" in name:
            (earlier if layer_number(name) < 12 else later).append(parameter)
    readout = list(model.readout.parameters()) + list(model.binding_head.parameters())
    groups = [
        {"params": earlier, "lr": earlier_lr, "name": "earlier_adapters"},
        {"params": later, "lr": later_lr, "name": "later_adapters"},
        {"params": readout, "lr": readout_lr, "name": "readout"},
    ]
    if any(not group["params"] for group in groups):
        raise ValueError("an expected trainable parameter group is empty")
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in groups for p in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("optimizer groups do not exactly cover trainable parameters")
    return groups
