"""Select the deployment checkpoint only from merged BF16 development evaluations."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from train_foundation_head import digest, write

from veyra.proper_learning import foundation_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-run", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "selection.json").exists():
        raise FileExistsError("selection is already frozen")
    completed = json.loads((args.backbone_run / "complete.json").read_text())
    if not completed["completed"]:
        raise ValueError("training incomplete")
    amendment = args.backbone_run / "midpoint-amendment.json"
    if amendment.exists():
        specification = json.loads(amendment.read_text())
        if specification["fixed_ratio"] != 0.5 or specification["final_or_calibration_used"]:
            raise ValueError("invalid parameter-average amendment")
        averaged = args.backbone_run / "head-final-midpoint"
        if not (averaged / "manifest.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "scripts/average_foundation_parameters.py",
                    "--first",
                    completed["candidates"][0],
                    "--second",
                    completed["candidates"][-1],
                    "--output",
                    str(averaged),
                ],
                check=True,
            )
        completed["candidates"].append(str(averaged))
    baseline = json.loads((args.head_run / "baseline.json").read_text())["dev"]
    protocol = {
        "candidates": completed["candidates"],
        "dataset_sha256": digest(args.records),
        "criterion": "Original release domain guards, then domain macro accuracy and NLL",
        "split": "dev",
        "final_or_calibration_used": False,
        "midpoint_amendment_sha256": digest(amendment) if amendment.exists() else None,
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("resumed selection protocol changed")
    else:
        write(protocol_path, protocol)
    results = []
    for index, checkpoint in enumerate(completed["candidates"]):
        output = args.output / f"candidate-{index}"
        if not (output / "evaluation.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_foundation.py",
                    "--checkpoint",
                    checkpoint,
                    "--records",
                    str(args.records),
                    "--split",
                    "dev",
                    "--output",
                    str(output),
                ],
                check=True,
            )
        evaluation = json.loads((output / "evaluation.json").read_text())
        if evaluation["protocol"]["weights_sha256"] != digest(
            Path(checkpoint) / "head.safetensors"
        ):
            raise ValueError("candidate differs from evaluated weights")
        results.append(
            {
                "checkpoint": checkpoint,
                "selection": foundation_selection(evaluation, baseline),
                "evaluation": str(output / "evaluation.json"),
            }
        )
    selected = max(results, key=lambda item: tuple(item["selection"]))
    if not selected["selection"][0]:
        raise ValueError("all merged candidates failed guardrails; preserve the prior release")
    write(
        args.output / "selection.json",
        {"candidates": results, "selected": selected, "final_or_calibration_used": False},
    )
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
