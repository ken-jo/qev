"""Resolution-elastic, option-permuted mix training on top of a released option checkpoint.

The released LoRA is merged at load; fresh LoRA (all text layers) plus the readout, condition
readout and binding heads are trained. Each request draws its own image budget from
``--pixels`` and, with ``--permute``, a fresh option order. Deltas are saved twice per epoch
and selected on the per-source development hold-out only, never on benchmarks.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from option_mix_common import (
    FIXED_ORDER_SOURCES,
    example_loss,
    head_modules,
    load_items,
    load_student,
    permute,
    record_examples,
    set_pixel_budget,
)
from safetensors.torch import save_file

from veyra.candidates import candidates_for


def pixel_list(value):
    return [int(part) for part in value.split(",") if part]


def source_caps(values):
    caps = {}
    for value in values:
        source, _, limit = value.partition("=")
        caps[source] = int(limit)
    return caps


@torch.no_grad()
def evaluate(model, examples, pixels, total_pixels):
    model.eval()
    counts = {}
    for index, example in enumerate(examples):
        images = len(example["request"].state.images)
        set_pixel_budget(model, pixels[index % len(pixels)], images, total_pixels)
        try:
            logits, _ = model(example["request"], example["root"])
        except ValueError:
            continue
        for name, question in example["request"].questions.items():
            keys = [candidate.key for candidate in candidates_for(question)]
            correct = keys[int(logits[name].argmax())] == example["gold"][name]
            total = counts.setdefault(example["kind"], [0, 0])
            total[0] += int(correct)
            total[1] += 1
    result = {kind: {"accuracy": right / n, "n": n} for kind, (right, n) in counts.items()}
    result["select"] = sum(value["accuracy"] for value in result.values()) / max(1, len(counts))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, required=True, help="Released checkpoint directory")
    parser.add_argument("--items", type=Path, nargs="+", required=True, help="items.jsonl files")
    parser.add_argument("--records", type=Path, help="Optional TrainingRecord replay JSONL")
    parser.add_argument("--replay-fraction", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, help="Continue from a saved delta")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--binding-lr", type=float, default=3e-4)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument(
        "--pixels", type=pixel_list, default=[], help="Comma list of per-image training caps"
    )
    parser.add_argument("--eval-pixels", type=pixel_list, default=[])
    parser.add_argument("--total-pixels", type=int, default=2097152)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--permute", action="store_true", help="Shuffle options per visit")
    parser.add_argument(
        "--fixed-order-sources", default=",".join(FIXED_ORDER_SOURCES), help="Never permuted"
    )
    parser.add_argument(
        "--source-cap", action="append", default=[], metavar="SOURCE=N", help="Per-source cap"
    )
    parser.add_argument("--dev-per-source", type=int, default=120)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--smoke", type=int, default=0, help="Use only N examples per split")
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("training output must be empty")
    if args.epochs < 1 or args.accumulation < 1 or not 0 <= args.replay_fraction <= 1:
        raise ValueError("training settings must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / "log.jsonl").open("a", encoding="utf-8")

    def say(**values):
        values["time"] = time.strftime("%H:%M:%S")
        print(json.dumps(values), flush=True)
        log.write(json.dumps(values) + "\n")
        log.flush()

    (args.output / "run.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    train, dev = load_items(args.items, args.dev_per_source, source_caps(args.source_cap))
    replay = []
    if args.records:
        replay = record_examples(args.records, "train")
        replay = rng.sample(replay, int(len(replay) * args.replay_fraction))
        dev += record_examples(args.records, "dev")
    if args.smoke:
        train, replay, dev = train[: 4 * args.smoke], replay[: args.smoke], dev[: args.smoke]
    say(event="data", train=len(train), replay=len(replay), dev=len(dev))

    model = load_student(
        args.base,
        args.device,
        args.cache_dir,
        args.max_tokens,
        args.initialize,
        offline=args.offline,
    )
    backbone = model.encoder.model
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    adapters = [p for name, p in backbone.named_parameters() if ".lora_" in name]
    heads = [p for module in head_modules(model)[:2] for p in module.parameters()]
    binding = [p for module in head_modules(model)[2:] for p in module.parameters()]
    for parameter in adapters + heads + binding:
        parameter.requires_grad_(True)
    groups = [
        {"params": adapters, "lr": args.adapter_lr},
        {"params": heads, "lr": args.head_lr},
        {"params": binding, "lr": args.binding_lr},
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    # Without explicit caps, evaluate at the checkpoint's own budget, not the last training draw.
    eval_pixels = args.eval_pixels or [model.encoder.config.image_max_pixels]
    if not args.skip_initial_eval:
        say(event="eval", tag="E0", **evaluate(model, dev, eval_pixels, args.total_pixels))

    data = train + replay
    fixed = set(filter(None, args.fixed_order_sources.split(",")))
    per_epoch = math.ceil(len(data) / args.accumulation)
    steps, step = args.epochs * per_epoch, 0
    warmup = math.ceil(0.03 * steps)
    half = max(1, per_epoch // 2)
    for epoch in range(1, args.epochs + 1):
        order = list(data)
        rng.shuffle(order)
        model.train()
        backbone.visual.eval()
        optimizer.zero_grad(set_to_none=True)
        started, seen, skipped, running = time.time(), 0, 0, None
        for index, example in enumerate(order, 1):
            if args.permute:
                example = permute(example, rng, fixed)
            if args.pixels:
                images = len(example["request"].state.images)
                set_pixel_budget(model, rng.choice(args.pixels), images, args.total_pixels)
            try:
                logits, _ = model(example["request"], example["root"])
            except ValueError:  # over the token budget after image expansion
                skipped += 1
            else:
                loss = example_loss(logits, example, args.smoothing) / args.accumulation
                loss.backward()
                value = float(loss.detach()) * args.accumulation
                running = value if running is None else 0.98 * running + 0.02 * value
                seen += 1
            if index % args.accumulation and index != len(order):
                continue
            if step < warmup:
                scale = (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, steps - warmup)
                scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            torch.nn.utils.clip_grad_norm_(adapters + heads + binding, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                elapsed = time.time() - started
                say(
                    event="step",
                    epoch=epoch,
                    step=step,
                    of=steps,
                    loss=round(running or 0.0, 4),
                    examples_per_second=round(seen / elapsed, 2),
                    skipped=skipped,
                )
            if step % half == 0 or step == steps:
                tag = f"E{step / per_epoch:.2f}"
                delta = {
                    name: p.detach().cpu().contiguous()
                    for name, p in model.named_parameters()
                    if p.requires_grad
                }
                save_file(delta, args.output / f"delta-{tag}.safetensors")
                result = evaluate(model, dev, eval_pixels, args.total_pixels)
                say(event="eval", tag=tag, **result)
                model.train()
                backbone.visual.eval()
    say(event="done")


if __name__ == "__main__":
    main()
