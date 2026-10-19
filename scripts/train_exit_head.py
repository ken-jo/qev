"""Distill early-exit heads from a full-depth option model.

One no-grad forward per request records the raw residual after each candidate exit layer at
the answer and internal condition positions, plus the full-depth option logits as teacher.
A small ``ExitHead`` per layer is then trained with KL to the teacher plus weighted gold
cross-entropy and saved as ``exit{L}.safetensors`` next to a ready ``early_exit`` block.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from option_mix_common import load_items, load_student, record_examples, set_pixel_budget
from safetensors.torch import save_file

from veyra.candidates import candidates_for
from veyra.exit_head import ExitHead
from veyra.option_model import option_prompt
from veyra.reasoning_workspace import RESULT_MARKER, condition_positions


@torch.no_grad()
def extract(model, examples, layers, pixels=None, total_pixels=2097152):
    """Rows of (features[len(layers), 2*hidden], teacher logits, gold index or -1, kind)."""
    language_model = model.encoder.model.language_model
    captured = {}

    def keep_inputs(_module, _args, kwargs):
        captured["ids"], captured["mask"] = kwargs["input_ids"], kwargs["attention_mask"]

    def keep_layer(layer):
        def hook(_module, _inputs, output):
            captured[layer] = output[0] if isinstance(output, tuple) else output

        return hook

    handles = [model.encoder.model.register_forward_pre_hook(keep_inputs, with_kwargs=True)]
    handles += [
        language_model.layers[layer - 1].register_forward_hook(keep_layer(layer))
        for layer in layers
    ]
    marker = model.encoder.processor.tokenizer.encode(RESULT_MARKER, add_special_tokens=False)
    rows = []
    try:
        model.eval()
        for example in examples:
            if pixels:
                images = len(example["request"].state.images)
                set_pixel_budget(model, pixels, images, total_pixels)
            try:
                teacher, _ = model(example["request"], example["root"])
            except ValueError:  # over the token budget
                continue
            mask = captured["mask"].cpu()
            last = (mask * torch.arange(mask.shape[1])).max(-1).values
            condition = condition_positions(captured["ids"].cpu(), mask, marker)
            # One prompt row per question (single decision view), in request order.
            for row, (name, question) in enumerate(example["request"].questions.items()):
                features = (
                    torch.stack(
                        [
                            torch.cat(
                                [
                                    captured[layer][row, last[row]],
                                    captured[layer][row, condition[row]],
                                ]
                            )
                            for layer in layers
                        ]
                    )
                    .float()
                    .cpu()
                )
                keys = [candidate.key for candidate in candidates_for(question)]
                gold = example["gold"].get(name)
                rows.append(
                    {
                        "features": features,
                        "positions": option_prompt(example["request"], question)[1],
                        "teacher": teacher[name].float().cpu(),
                        "gold": keys.index(gold) if gold in keys else -1,
                        "kind": example["kind"],
                    }
                )
    finally:
        for handle in handles:
            handle.remove()
    return rows


def distill(rows, index, hidden, args):
    head = ExitHead(hidden).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), args.learning_rate, weight_decay=0.01)
    order = list(range(len(rows)))
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            features = torch.stack([rows[i]["features"][index] for i in batch]).to(args.device)
            logits = head.f(head.n(features))
            loss = 0
            for j, i in enumerate(batch):
                row = rows[i]
                log_probs = logits[j, row["positions"]].log_softmax(-1)
                target = (row["teacher"].to(args.device) / args.temperature).softmax(-1)
                loss = loss + (target * (target.clamp_min(1e-9).log() - log_probs)).sum()
                if row["gold"] >= 0:
                    loss = loss + args.gold_weight * -log_probs[row["gold"]]
            optimizer.zero_grad()
            (loss / len(batch)).backward()
            optimizer.step()
    return head.eval()


@torch.no_grad()
def score(head, rows, index, device):
    counts = {}
    for row in rows:
        features = row["features"][index][None].to(device)
        exit_choice = int(head.f(head.n(features))[0, row["positions"]].argmax())
        full_choice = int(row["teacher"].argmax())
        total = counts.setdefault(row["kind"], [0, 0, 0, 0])
        total[0] += exit_choice == row["gold"]
        total[1] += full_choice == row["gold"]
        total[2] += exit_choice == full_choice
        total[3] += 1
    return {
        kind: {"exit": e / n, "full": f / n, "agreement": a / n, "n": n}
        for kind, (e, f, a, n) in counts.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, required=True, help="Released checkpoint directory")
    parser.add_argument("--delta", type=Path, help="Optional delta from train_option_mix.py")
    parser.add_argument("--items", type=Path, nargs="+", required=True, help="items.jsonl files")
    parser.add_argument("--records", type=Path, help="Optional TrainingRecord replay JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="12,14,16,18,20", help="Comma list of exit layers")
    parser.add_argument("--train-count", type=int, default=12000)
    parser.add_argument("--replay-ratio", type=float, default=0.25, help="Replay per train item")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gold-weight", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pixels", type=int, help="Per-image budget during extraction")
    parser.add_argument("--total-pixels", type=int, default=2097152)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    layers = [int(value) for value in args.layers.split(",") if value]
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("exit head output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train, dev = load_items(args.items)
    train = rng.sample(train, min(args.train_count, len(train)))
    if args.records:
        replay = record_examples(args.records, "train")
        train += rng.sample(replay, min(int(len(train) * args.replay_ratio), len(replay)))
        dev += record_examples(args.records, "dev")
    model = load_student(
        args.base, args.device, args.cache_dir, args.max_tokens, args.delta, offline=args.offline
    )
    total = len(model.encoder.model.language_model.layers)
    if not all(1 <= layer < total for layer in layers):
        raise ValueError(f"exit layers must be in 1..{total - 1}")
    train_rows = extract(model, train, layers, args.pixels, args.total_pixels)
    dev_rows = extract(model, dev, layers, args.pixels, args.total_pixels)
    print(json.dumps({"extracted": {"train": len(train_rows), "dev": len(dev_rows)}}), flush=True)
    hidden = model.encoder.model.language_model.config.hidden_size
    summary = {}
    for index, layer in enumerate(layers):
        head = distill(train_rows, index, hidden, args)
        path = args.output / f"exit{layer}.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, path)
        summary[layer] = {
            "early_exit": {
                "layer": layer,
                "head": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "development": score(head, dev_rows, index, args.device),
        }
        print(json.dumps({"layer": layer, **summary[layer]}), flush=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
