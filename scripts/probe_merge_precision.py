"""Matched development comparison of adapter merge arithmetic; legacy default is unchanged."""

import argparse
import gc
import json
from pathlib import Path

import torch
from train_option import evaluate

from veyra.data import read_records
from veyra.option_model import OptionModel

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--records", type=Path, default=Path("data/policy-v3/records.jsonl"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("use a new report path")
records = [record for record in read_records(args.records) if record.split == "dev"]
report = {"split": "dev", "checkpoint": str(args.checkpoint), "merge_accumulation": {}}
for accumulation in ("base", "float32"):
    model = OptionModel.load(
        args.checkpoint, local_files_only=True, merge_accumulation=accumulation
    )
    metrics = evaluate(model, records, args.records.parent)
    report["merge_accumulation"][accumulation] = metrics
    print(json.dumps({"merge_accumulation": accumulation, "metrics": metrics}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
