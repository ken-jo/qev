"""Shared data, loss and model setup for the mix-training and exit-distillation recipes.

Multiple-choice items are JSON lines with ``id``, ``source``, ``question``, ``options`` (2-16
strings), ``gold`` (an option letter) and ``images`` (paths, relative to the items file or
absolute). Optional replay uses the public ``TrainingRecord`` format from ``veyra.data``.
"""

import json
import random
import zlib
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.option_model import OptionModel, install_adapters
from veyra.schema import DecisionRequest

LETTERS = "ABCDEFGHIJKLMNOP"
# Options that name letters or boxes drawn in the image must keep their order.
FIXED_ORDER_SOURCES = ("mind2web", "mind2web_k", "options_in_image")


def item_request(item):
    """The single-question request shape that evaluation harnesses send at inference time."""
    criteria = {}
    for index, option in enumerate(item["options"]):
        value = option.strip() or LETTERS[index]
        while value in criteria.values():
            value = f"{value} ({LETTERS[index]})"
        criteria[LETTERS[index]] = value[:8000]
    instructions = item["question"].strip()[:8000] or "Choose the correct option."
    return DecisionRequest.model_validate(
        {
            "state": {"text": "", "images": [{"path": path} for path in item["images"]]},
            "questions": {
                "answer": {"type": "choice", "instructions": instructions, "criteria": criteria}
            },
        }
    )


def item_example(item):
    if not 2 <= len(item["options"]) <= 16 or item["gold"] not in LETTERS[: len(item["options"])]:
        raise ValueError(f"item {item['id']} needs 2-16 options and a valid gold letter")
    return {
        "kind": "item-" + item["source"],
        "request": item_request(item),
        "root": None,
        "targets": None,
        "gold": {"answer": item["gold"]},
        "item": item,
    }


def load_items(paths, dev_per_source=120, caps=None, seed=0):
    """Split each items file into train and a stable ~5% development hold-out (by ID hash)."""
    train, dev = [], []
    for path in paths:
        path = Path(path)
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                images = [
                    Path(p) if Path(p).is_absolute() else path.parent / p for p in item["images"]
                ]
                items.append(dict(item, images=[str(p.resolve()) for p in images]))
        limit = (caps or {}).get(path.parent.name)
        if limit is not None and len(items) > limit:
            items = random.Random(zlib.crc32(path.parent.name.encode()) ^ seed).sample(items, limit)
        held_out = min(dev_per_source, len(items) // 20)
        for item in items:
            example = item_example(item)
            if held_out and zlib.crc32(item["id"].encode()) % 20 == 0:
                dev.append(example)
                held_out -= 1
            else:
                train.append(example)
    return train, dev


def record_examples(path, split):
    """Soft-target replay examples from a public records file."""
    path = Path(path)
    return [
        {
            "kind": "record-" + record.family,
            "request": record.request,
            "root": path.parent,
            "targets": record.targets,
            "gold": {name: max(target, key=target.get) for name, target in record.targets.items()},
            "item": None,
        }
        for record in read_records(path)
        if record.split == split
    ]


def permute(example, rng, fixed_sources=FIXED_ORDER_SOURCES):
    """Fresh option order per visit, so the readout binds to option content, not letter position."""
    item = example["item"]
    if item is None or item["source"] in fixed_sources:
        return example
    order = list(range(len(item["options"])))
    rng.shuffle(order)
    gold = LETTERS[order.index(LETTERS.index(item["gold"]))]
    shuffled = dict(item, options=[item["options"][i] for i in order], gold=gold)
    return dict(example, request=item_request(shuffled), gold={"answer": gold}, item=shuffled)


def set_pixel_budget(model, pixels, images, total_pixels):
    """Per-image cap for one request; multi-image requests share ``total_pixels``."""
    processor = model.encoder.processor.image_processor
    shortest = processor.size["shortest_edge"]
    per_image = min(pixels, total_pixels // max(1, images))
    processor.size = {"shortest_edge": shortest, "longest_edge": max(shortest, per_image)}


def example_loss(logits, example, smoothing=0.05):
    """Soft cross-entropy on record targets; label-smoothed gold cross-entropy on items."""
    losses = []
    for name, question in example["request"].questions.items():
        keys = [candidate.key for candidate in candidates_for(question)]
        log_probs = logits[name].float().log_softmax(-1)
        target = (example["targets"] or {}).get(name)
        if target is not None:
            weights = torch.tensor([target[key] for key in keys], device=log_probs.device)
            losses.append(-(weights * log_probs).sum())
        else:
            gold = keys.index(example["gold"][name])
            losses.append((1 - smoothing) * -log_probs[gold] + smoothing * -log_probs.mean())
    return torch.stack(losses).sum()


def load_student(
    base, device, cache_dir, max_tokens, initialize=None, rank=8, alpha=16.0, offline=False
):
    """Released checkpoint with its LoRA merged, plus fresh adapters on every text layer."""
    model = OptionModel.load(base, device=device, cache_dir=cache_dir, local_files_only=offline)
    if model.decision_views != 1 or not model.reasoning_slots:
        raise ValueError("the recipe expects one decision view and internal condition slots")
    model.encoder.config = replace(model.encoder.config, max_total_tokens=max_tokens)
    language_model = model.encoder.model.language_model
    model.adapter_names = install_adapters(
        model.encoder.model, len(language_model.layers), rank, alpha
    )
    if initialize is not None:
        state = load_file(initialize)
        unknown = set(state) - set(dict(model.named_parameters()))
        if unknown:
            raise ValueError(f"initialization keys not in model: {sorted(unknown)[:5]}")
        model.load_state_dict(state, strict=False)
    return model


def head_modules(model):
    return [model.readout, model.condition_readout] + (
        [model.binding_head] if model.binding_rank else []
    )
