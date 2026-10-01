"""Select activation storage without changing adapter placement or inference behavior."""


def configure_checkpointing(model, strategy):
    if strategy not in {"none", "earlier12", "all24"}:
        raise ValueError("unknown depth checkpointing strategy")
    layers = model.encoder.model.language_model.layers
    if len(layers) != 24 or model.adaptation["layers"] != 24:
        raise ValueError("execution comparison requires all 24 adapter layers")
    selected = {
        "none": set(),
        "earlier12": set(range(12)),
        "all24": set(range(24)),
    }[strategy]
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index in selected
    return sorted(selected)
