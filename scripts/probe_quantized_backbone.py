"""Independent, development-only 4B NF4 baseline; does not change published checkpoints."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5Model

from veyra.backbone import QwenEncoder
from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.option_model import OptionConfig, OptionModel
from veyra.probability import Calibration, typed_answer
from veyra.schema import DecisionRequest

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/policy-v3/records.jsonl"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("probe reports are immutable; use a new output path")
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        MODEL,
        revision=REVISION,
        cache_dir=".cache/huggingface",
        local_files_only=True,
        trust_remote_code=False,
    )
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}
    base = (
        Qwen3_5Model.from_pretrained(
            MODEL,
            revision=REVISION,
            cache_dir=".cache/huggingface",
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["visual"],
            ),
        )
        .eval()
        .requires_grad_(False)
    )
    encoder = QwenEncoder(base, processor, OptionConfig())
    # Reuse exactly the current option input path, with a dimension-independent readout.
    probe = SimpleNamespace(encoder=encoder, adapter_names=[])
    ids = [processor.tokenizer.encode(chr(65 + i), add_special_tokens=False) for i in range(16)]
    assert all(len(value) == 1 for value in ids)
    readout = base.language_model.embed_tokens.weight[
        torch.tensor([value[0] for value in ids], device="cuda")
    ].float()
    load_seconds = time.perf_counter() - started
    records = [r for r in read_records(args.records) if r.split == "dev"]

    def run(request):
        states, mappings, tokens = OptionModel.encode(probe, request, args.records.parent)
        values = states @ readout.T
        logits = {
            name: values[i, positions] for i, (name, positions) in enumerate(mappings.items())
        }
        answer = {
            name: typed_answer(request.questions[name], value, Calibration())
            for name, value in logits.items()
        }
        return logits, answer, tokens

    for modality in ("text", "image"):
        warm = next(r.request for r in records if r.family.startswith(modality + "_"))
        for _ in range(3):
            run(warm)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for index, record in enumerate(records):
        raw = record.request.model_dump_json()
        torch.cuda.synchronize()
        started = time.perf_counter()
        request = DecisionRequest.from_json(raw)
        logits, answer, tokens = run(request)
        json.dumps(answer)
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - started)
        for name, value in logits.items():
            question = request.questions[name]
            candidates = candidates_for(question)
            targets = torch.tensor(
                [record.targets[name][c.key] for c in candidates], device=value.device
            )
            rows.append(
                {
                    "id": record.id,
                    "family": record.family,
                    "type": question.type,
                    "correct": float(targets[int(value.argmax())]),
                    "nll": float(-(targets * value.log_softmax(-1)).sum()),
                    "milliseconds": elapsed,
                    "input_tokens": tokens,
                }
            )
        if (index + 1) % 100 == 0:
            print(f"evaluated {index + 1}/{len(records)} development requests", flush=True)

    def summarize(selected):
        return {
            "questions": len(selected),
            "accuracy": np.mean([r["correct"] for r in selected]),
            "nll": np.mean([r["nll"] for r in selected]),
            "p95_ms": np.percentile([r["milliseconds"] for r in selected], 95),
        }

    report = {
        "model": MODEL,
        "revision": REVISION,
        "trained": False,
        "split": "dev",
        "dataset_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "evaluated_subset_sha256": hashlib.sha256(
            "".join(r.model_dump_json() + "\n" for r in records).encode()
        ).hexdigest(),
        "quantization": "bitsandbytes 0.50.2 NF4 double quant; BF16 compute; vision excluded",
        "load_seconds": load_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "timing_scope": (
            "JSON parse, image decode, option encoding, Qwen, readout, typed JSON serialize"
        ),
        "feature_cache": False,
        "overall": summarize(rows),
        "by_modality": {
            kind: summarize([r for r in rows if r["family"].startswith(kind + "_")])
            for kind in ("text", "image")
        },
        "by_family": {
            kind: summarize([r for r in rows if r["family"] == kind])
            for kind in sorted({r["family"] for r in rows})
        },
        "predictions": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2)
    )


if __name__ == "__main__":
    main()
