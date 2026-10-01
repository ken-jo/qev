"""Count stock Laya tokenizer truncations without loading weights or evaluating labels."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_laya_reference import official_records
from train_foundation_head import write

from veyra.data import read_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=int, choices=[0, 1], required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-len", type=int)
    args = parser.parse_args()
    if bool(args.records) == bool(args.official) or args.output.exists():
        raise ValueError("use one input corpus and a new audit report")
    reference = json.loads(Path("runs/foundation-v11-laya/sources.json").read_text())[
        args.reference
    ]
    sys.path.insert(0, str(Path(".cache/laya-sdk").resolve()))
    from laya.agent import Agent, _load_tokenizer
    from laya.common import render_options

    folder = Path(reference["path"])
    config = json.loads((folder / "rl_agent_config.json").read_text())
    if args.max_len is not None:
        if args.max_len < 1:
            raise ValueError("max length must be positive")
        config["max_len"] = args.max_len
    tokenizer = _load_tokenizer(str(folder / "tokenizer"), config)
    records = (
        official_records(args.official)
        if args.official
        else [
            r
            for r in read_records(args.records)
            if r.split == "test" and not r.request.state.images
        ]
    )
    original_questions = {}
    if args.official:
        for line in args.official.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            for name, question in json.loads(case["questions"]).items():
                original_questions[case["id"] + ":" + name] = question
    buckets = defaultdict(Counter)
    for record in records:
        for question in record.request.questions.values():
            q = Agent._to_internal(original_questions.get(record.id, question.model_dump()))
            instruction = tokenizer(
                q["t"] + " question: " + q["ins"].replace(tokenizer.mask_token, " "),
                add_special_tokens=False,
            )["input_ids"]
            original = [
                1
                + len(
                    tokenizer(
                        " " + option.replace(tokenizer.mask_token, " "), add_special_tokens=False
                    )["input_ids"]
                )
                for option in render_options(q)
            ]
            kept = [min(49, size) for size in original]
            budget = config.get("head_max_len", 192) - sum(kept)
            if budget < 16:
                limit = max(4, (config.get("head_max_len", 192) - 16) // len(kept))
                kept = [min(size, limit) for size in kept]
                budget = config.get("head_max_len", 192) - sum(kept)
            head = min(len(instruction), max(8, budget))
            room = max(0, config.get("max_len", 512) - head - sum(kept) - 4)
            state_size = len(
                tokenizer(
                    record.request.state.text.replace(tokenizer.mask_token, " "),
                    add_special_tokens=False,
                )["input_ids"]
            )
            events = {
                "questions": 1,
                "instruction_truncated": int(head < len(instruction)),
                "options_truncated": int(original != kept),
                "state_truncated": int(state_size > room),
            }
            events["any_truncated"] = int(any(events[k] for k in events if k != "questions"))
            buckets["overall"].update(events)
            buckets[record.family].update(events)
    write(
        args.output,
        {
            "reference": reference["repo"],
            "revision": reference["revision"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "max_len": config.get("max_len", 512),
            "head_max_len": config.get("head_max_len", 192),
            "option_token_cap": 48,
            "label_or_prediction_inspected": False,
            "counts": {key: dict(value) for key, value in buckets.items()},
        },
    )
    print(json.dumps({key: dict(value) for key, value in buckets.items()}), flush=True)


if __name__ == "__main__":
    main()
