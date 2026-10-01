"""Freeze a development-selected epoch together with its complete run provenance."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reason", required=True)
parser.add_argument("--discarded-partial-epoch", type=int)
parser.add_argument("--merge-accumulation", choices=["base", "float32"])
args = parser.parse_args()
manifest = json.loads((args.source / "manifest.json").read_text())
run = json.loads((args.run / "run.json").read_text())
weights = args.source / "head.safetensors"
if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
    raise ValueError("source checksum mismatch")
if args.output.exists() and any(args.output.iterdir()):
    raise FileExistsError("selected checkpoint output must be empty")
manifest["decision_views"] = 1
manifest["merge_accumulation"] = args.merge_accumulation or manifest.get(
    "merge_accumulation", "base"
)
manifest["training"].update(
    intermediate=False,
    selected_epoch=manifest["training"].get("selected_epoch", manifest["training"].get("epochs")),
    selection_split="dev",
    selection_metric=run["arguments"]["selection"],
    selection_reason=args.reason,
    discarded_partial_epoch=args.discarded_partial_epoch,
    run_arguments=run["arguments"],
    source_sha256=run["source_sha256"],
    final_evaluated_at_selection=False,
)
args.output.mkdir(parents=True, exist_ok=True)
shutil.copyfile(weights, args.output / "head.safetensors")
(args.output / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
)
print(json.dumps({"selected": str(args.output), "weights_sha256": manifest["weights_sha256"]}))
