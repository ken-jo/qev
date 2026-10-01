"""Select only from the declared candidates using merged BF16 development inference."""

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_foundation_head import write
from train_workflow_backbone import assess

from veyra.data import read_records
from veyra.option_model import OptionModel
from veyra.workflow_learning import annotate, selection_key


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = Path("runs/workflow-v12-selection")
    if (output / "selection.json").exists():
        raise FileExistsError("development selection is frozen")
    config_path = Path("configs/workflow-release-v12.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path("data/features-workflow-v12")
    tensors = load_file(root / "features.safetensors")
    rows = json.loads((root / "records.json").read_text(encoding="utf-8"))
    data_path = Path("data/workflow-v12/records.jsonl")
    records = {r.id: r for r in read_records(data_path) if r.split in {"train", "dev"}}
    annotate(rows, records.values())
    dev = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "dev"])
    head = json.loads(Path("runs/workflow-v12-head/selection.json").read_text(encoding="utf-8"))[
        "selected"
    ]
    candidates = [Path(head["checkpoint"])]
    for arm in ("low", "high"):
        complete = json.loads(
            Path(f"runs/workflow-v12-backbone/{arm}/complete.json").read_text(encoding="utf-8")
        )
        if not complete["completed"]:
            raise ValueError("unfinished backbone experiment")
        candidates.extend(Path(p) for p in complete["candidates"])
    write(
        output / "protocol.json",
        {
            "release_protocol_sha256": digest(config_path),
            "records_sha256": digest(data_path),
            "candidates": [str(p) for p in candidates],
            "source_sha256": digest(Path(__file__)),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "final_or_calibration_used": False,
        },
    )
    torch.set_num_threads(4)
    baseline_path = Path(config["baseline"])
    if (
        digest(baseline_path / "head.safetensors") != config["baseline_weights_sha256"]
        or digest(baseline_path / "manifest.json") != config["baseline_manifest_sha256"]
    ):
        raise ValueError("frozen baseline changed")
    baseline_report = output / "merged-baseline-dev.json"
    if baseline_report.exists():
        baseline_result = json.loads(baseline_report.read_text(encoding="utf-8"))
        if (
            baseline_result["weights_sha256"] != config["baseline_weights_sha256"]
            or baseline_result["manifest_sha256"] != config["baseline_manifest_sha256"]
            or baseline_result["records_sha256"] != digest(data_path)
        ):
            raise ValueError("resumed baseline evidence mismatch")
    else:
        model = OptionModel.load(baseline_path, local_files_only=True, merge=True)
        baseline_result = {
            "weights_sha256": config["baseline_weights_sha256"],
            "manifest_sha256": config["baseline_manifest_sha256"],
            "records_sha256": digest(data_path),
            "metrics": assess(
                model,
                records,
                data_path.parent,
                rows,
                tensors,
                dev,
                model.calibration.temperatures,
            ),
        }
        write(baseline_report, baseline_result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    baseline = baseline_result["metrics"]
    results = []
    for index, path in enumerate(candidates):
        result_path = output / f"merged-dev-{index}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["weights_sha256"] != digest(path / "head.safetensors") or result[
                "manifest_sha256"
            ] != digest(path / "manifest.json"):
                raise ValueError("candidate changed during resumed selection")
            result["selection"] = selection_key(result["metrics"], baseline)
        else:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            model = OptionModel.load(path, local_files_only=True, merge=True)
            metrics = assess(
                model,
                records,
                data_path.parent,
                rows,
                tensors,
                dev,
                manifest["calibration"]["temperatures"],
            )
            result = {
                "checkpoint": str(path),
                "weights_sha256": digest(path / "head.safetensors"),
                "manifest_sha256": digest(path / "manifest.json"),
                "metrics": metrics,
                "selection": selection_key(metrics, baseline),
            }
            write(result_path, result)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result)
        print(
            json.dumps(
                {"candidate": index, "checkpoint": str(path), "selection": result["selection"]}
            ),
            flush=True,
        )
    selected = max(results, key=lambda r: tuple(r["selection"]))
    result = {
        "selected": selected,
        "weights_sha256": selected["weights_sha256"],
        "manifest_sha256": selected["manifest_sha256"],
        "baseline_merged_dev": str(baseline_report),
        "baseline_merged_dev_sha256": digest(baseline_report),
        "records_sha256": digest(data_path),
        "release_protocol_sha256": digest(config_path),
        "eligible": bool(selected["selection"][0]),
        "required_priorities": config["required_priorities"],
        "final_or_calibration_used": False,
    }
    write(output / "selection.json", result)
    print(
        json.dumps({"selected": selected["checkpoint"], "eligible": result["eligible"]}), flush=True
    )


if __name__ == "__main__":
    main()
