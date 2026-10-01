"""Development-only comparison of option letters and native boolean answer tokens."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.option_model import OptionModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/policy-v3/records.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--family-prefix", default="")
    parser.add_argument("--limit-per-family", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new report path")
    if args.threads:
        torch.set_num_threads(args.threads)
    model = OptionModel.load(args.checkpoint, local_files_only=True, device=args.device)
    if model.decision_views != 1:
        raise ValueError("this comparison requires a one-view checkpoint")
    records = [
        record
        for record in read_records(args.records)
        if record.split == "dev"
        and record.family.startswith(args.family_prefix)
        and all(q.type == "noul" for q in record.request.questions.values())
    ]
    if args.limit_per_family:
        counts, selected = {}, []
        for record in records:
            counts[record.family] = counts.get(record.family, 0) + 1
            if counts[record.family] <= args.limit_per_family:
                selected.append(record)
        records = selected
    if not records:
        raise ValueError("no development boolean records selected")
    report = {
        "split": "dev",
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "precision": "float32" if args.device == "cpu" else "bfloat16",
        "limit_per_family": args.limit_per_family,
        "family_prefix": args.family_prefix,
        "variants": {},
    }
    for variant, words in (
        ("trained_option_letters", None),
        ("lowercase_boolean", ("false", "true")),
        ("titlecase_boolean", ("False", "True")),
        ("no_yes", ("No", "Yes")),
    ):
        if words:
            ids = [
                model.encoder.processor.tokenizer.encode(w, add_special_tokens=False) for w in words
            ]
            if any(len(value) != 1 for value in ids):
                raise ValueError("answer words must be single tokens")
            readout = model.encoder.model.language_model.embed_tokens.weight[
                torch.tensor([value[0] for value in ids], device=model.encoder.device)
            ].float()

            def native_prompt(request, question, rotation=0):
                assert question.type == "noul" and rotation == 0
                lines = ["Evidence:", request.state.text, "Question:", question.instructions]
                for word, candidate in zip(words, candidates_for(question), strict=True):
                    lines.append(f"{word}: {candidate.description}")
                lines.append(f"Answer with only {words[0]} or {words[1]}.")
                return "\n".join(lines), [0, 1]

        rows = []
        for index, record in enumerate(records):
            if words:
                with patch("veyra.option_model.option_prompt", native_prompt):
                    states, _, _ = model.encode(record.request, args.records.parent)
                values = states @ readout.T
                logits = {name: values[i] for i, name in enumerate(record.request.questions)}
            else:
                logits, _ = model(record.request, args.records.parent)
            for name, value in logits.items():
                targets = torch.tensor(
                    [record.targets[name][key] for key in ("false", "true")], device=value.device
                )
                rows.append(
                    {
                        "id": record.id,
                        "family": record.family,
                        "correct": float(targets[int(value.argmax())]),
                        "nll": float(-(targets * value.log_softmax(-1)).sum()),
                        "probabilities": value.softmax(-1).tolist(),
                    }
                )
            if (index + 1) % 8 == 0:
                print(f"{variant}: {index + 1}/{len(records)}", flush=True)

        def summary(selected):
            return {
                "questions": len(selected),
                "accuracy": sum(r["correct"] for r in selected) / len(selected),
                "nll": sum(r["nll"] for r in selected) / len(selected),
            }

        result = {
            "overall": summary(rows),
            "by_modality": {
                kind: summary([r for r in rows if r["family"].startswith(kind + "_")])
                for kind in ("text", "image")
                if any(r["family"].startswith(kind + "_") for r in rows)
            },
            "predictions": rows,
        }
        report["variants"][variant] = result
        print(json.dumps({"variant": variant, "overall": result["overall"]}), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
