"""Verify a saved workspace checkpoint on GPU without evaluating any final records."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from veyra.data import read_records
from veyra.option_model import OptionModel

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("smoke reports are immutable")
model = OptionModel.load(args.checkpoint, local_files_only=True)
records = [r for r in read_records(args.records) if r.split == "dev"]
probes = [next(r for r in records if r.family.startswith(kind + "_")) for kind in ("text", "image")]
calls, rows = [], []
handle = model.encoder.model.register_forward_hook(lambda *_args: calls.append(1))
with torch.inference_mode():
    for record in probes:
        ordinary, tokens = model(record.request, args.records.parent)
        calls.clear()
        with_auxiliary, auxiliary_tokens, auxiliary = model.forward_with_condition(
            record.request, args.records.parent
        )
        difference = max(float((ordinary[k] - with_auxiliary[k]).abs().max()) for k in ordinary)
        if len(calls) != 1 or difference != 0 or tokens != auxiliary_tokens:
            raise ValueError("workspace changed the decision path or required extra backbone calls")
        if not all(torch.isfinite(value).all() for value in auxiliary.values()):
            raise ValueError("nonfinite internal condition logits")
        response = model.predict(record.request, args.records.parent)
        if response["usage"]["output_tokens"] != 0:
            raise ValueError("workspace inference must not generate text")
        rows.append(
            {
                "id": record.id,
                "family": record.family,
                "backbone_calls_with_auxiliary": 1,
                "decision_logit_difference": difference,
                "input_tokens": tokens,
                "output_tokens": 0,
                "finite_condition_logits": True,
            }
        )
handle.remove()
report = {
    "passed": True,
    "scope": "GPU checkpoint reload and single-forward plumbing, not quality",
    "reasoning_slots": model.reasoning_slots,
    "weights_sha256": hashlib.sha256(
        (args.checkpoint / "head.safetensors").read_bytes()
    ).hexdigest(),
    "probes": rows,
}
args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
