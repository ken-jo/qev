"""Create a CPU-only, uncalibrated parameter average for later development evaluation."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from veyra.average_adapters import average_states
from veyra.probability import Calibration

parser = argparse.ArgumentParser()
parser.add_argument("--sources", type=Path, nargs="+", required=True)
parser.add_argument("--weights", type=float, nargs="+")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists() and any(args.output.iterdir()):
    raise FileExistsError("checkpoint output must be empty")
manifests, states = [], []
for source in args.sources:
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("format_version") != 2 or manifest.get("architecture") != "option_readout":
        raise ValueError("only option-readout checkpoints can be averaged")
    if manifest.get("reasoning_slots", 0):
        raise ValueError("workspace checkpoint averaging is not implemented")
    if (
        not manifest["training"].get("completed")
        or manifest["training"].get("optimizer_steps", 0) < 1
    ):
        raise ValueError("all source training checkpoints must be complete")
    weights_file = source / "head.safetensors"
    if hashlib.sha256(weights_file.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("source checkpoint checksum mismatch")
    if manifests:
        for field in ("backbone", "encoder", "decision_views"):
            if manifest.get(field, 1 if field == "decision_views" else None) != manifests[0].get(
                field, 1 if field == "decision_views" else None
            ):
                raise ValueError(f"source {field} mismatch")
        if manifest["adaptation"]["layers"] != manifests[0]["adaptation"]["layers"]:
            raise ValueError("source adapted layer counts differ")
    manifests.append(manifest)
    states.append(load_file(weights_file))
weights = args.weights or [1.0] * len(states)
scales = [m["adaptation"]["alpha"] / m["adaptation"]["rank"] for m in manifests]
averaged = average_states(states, scales, weights)
rank = sum(m["adaptation"]["rank"] for m in manifests)
for name, value in averaged.items():
    if name.endswith(".lora_a") and value.shape[0] != rank:
        raise ValueError("source manifest rank does not match tensors")
result = copy.deepcopy(manifests[0])
result["adaptation"].update(rank=rank, alpha=float(rank))
result["merge_accumulation"] = "float32"
result["calibration"] = Calibration().to_dict()
result["training"] = {
    "completed": True,
    "optimizer_steps": max(m["training"]["optimizer_steps"] for m in manifests),
    "selection_split": "dev",
    "intermediate": True,
    "composition": {
        "method": "weighted_mean_of_adapter_updates_and_readout",
        "new_optimizer_steps": 0,
        "normalized_weights": [weight / sum(weights) for weight in weights],
        "source_checkpoints": [str(path) for path in args.sources],
        "source_manifests": manifests,
        "source_sha256": hashlib.sha256(
            Path("src/veyra/average_adapters.py").read_bytes()
        ).hexdigest(),
    },
}
args.output.mkdir(parents=True, exist_ok=True)
temporary = args.output / "head.safetensors.tmp"
save_file(averaged, temporary)
temporary.replace(args.output / "head.safetensors")
result["weights_sha256"] = hashlib.sha256(
    (args.output / "head.safetensors").read_bytes()
).hexdigest()
temporary_manifest = args.output / "manifest.json.tmp"
temporary_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
temporary_manifest.replace(args.output / "manifest.json")
print(
    json.dumps(
        {"output": str(args.output), "rank": rank, "weights_sha256": result["weights_sha256"]}
    )
)
