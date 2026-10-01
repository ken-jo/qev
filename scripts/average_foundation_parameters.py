"""Construct one declared parameter midpoint; it adds no inference-time ensemble."""

import argparse
import copy
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
from train_foundation_head import digest, write

from veyra.probability import Calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("parameter-average output must be empty")
    parents = [
        json.loads((path / "manifest.json").read_text()) for path in (args.first, args.second)
    ]
    for key in (
        "format_version",
        "architecture",
        "backbone",
        "encoder",
        "adaptation",
        "decision_views",
        "reasoning_slots",
        "binding_head",
    ):
        if parents[0][key] != parents[1][key]:
            raise ValueError("incompatible parent " + key)
    for path, parent in zip((args.first, args.second), parents, strict=True):
        if digest(path / "head.safetensors") != parent["weights_sha256"]:
            raise ValueError("parent weights changed")
    first, second = [load_file(path / "head.safetensors") for path in (args.first, args.second)]
    if set(first) != set(second) or any(first[key].shape != second[key].shape for key in first):
        raise ValueError("parent parameter keys/shapes differ")
    args.output.mkdir(parents=True)
    averaged = {key: ((first[key].float() + second[key].float()) / 2).contiguous() for key in first}
    save_file(averaged, args.output / "head.safetensors")
    manifest = copy.deepcopy(parents[1])
    manifest["weights_sha256"] = digest(args.output / "head.safetensors")
    manifest["calibration"] = Calibration().to_dict()
    manifest["training"]["parameter_average"] = {
        "parents": [parent["weights_sha256"] for parent in parents],
        "weights": [0.5, 0.5],
        "scope": (
            "Average stored LoRA factors and readouts, not probabilities or merged base matrices"
        ),
        "direct_optimizer_steps": 0,
        "ratio_selected_using_final": False,
        "source_sha256": digest(Path(__file__)),
    }
    manifest["training"].pop("selected_dev", None)
    manifest["training"].pop("selection", None)
    write(args.output / "manifest.json", manifest)
    print(
        json.dumps({"checkpoint": str(args.output), "weights_sha256": manifest["weights_sha256"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
