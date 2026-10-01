"""Assess direct prerequisites on a declared development-only sample before training."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file
from train_foundation_head import write
from workflow_skill_common import assess_skills

from veyra.data import read_records
from veyra.option_model import OptionModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path("runs/workflow-v12-skill-diagnostic")
    if output.exists():
        raise FileExistsError("skill diagnosis is immutable")
    data = Path("data/workflow-v12-skills")
    audit = json.loads((data / "audit.json").read_text())
    if not audit["passed"] or audit["calibration_or_final_used"]:
        raise ValueError("skill data audit failed")
    for name, expected in audit["files_sha256"].items():
        if digest(data / name) != expected:
            raise ValueError("skill data changed")
    chosen = json.loads((data / "diagnostic-ids.json").read_text())
    all_records = {r.id: r for r in read_records(data / "dev.jsonl")}
    records = [all_records[key] for key in chosen]
    if len(records) != 1200 or any(r.split != "dev" for r in records):
        raise ValueError("invalid skill diagnosis sample")
    checkpoint = Path("runs/workflow-v12-workspace-study/primitive/epoch-2/checkpoint")
    if (
        digest(checkpoint / "head.safetensors")
        != "4bb3c3187ec37e6ad7f6634c8aeec9bafbbb90ebea89ebae0d26f7c844a87c72"
    ):
        raise ValueError("diagnosis checkpoint changed")
    write(
        output / "protocol.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "weights_sha256": digest(checkpoint / "head.safetensors"),
            "manifest_sha256": digest(checkpoint / "manifest.json"),
            "data_audit_sha256": digest(data / "audit.json"),
            "source_sha256": {
                str(p): digest(p)
                for p in (
                    Path(__file__),
                    Path("scripts/workflow_skill_common.py"),
                    Path("scripts/train_workflow_backbone.py"),
                    Path("src/veyra/option_model.py"),
                )
            },
            "scope": (
                "Exploratory direct prerequisite diagnosis; fixed before training; merged BF16; "
                "no policy fitting or final/calibration use."
            ),
            "release_allowed": False,
        },
    )
    torch.set_num_threads(4)
    model = OptionModel.load(checkpoint, local_files_only=True, merge=True)
    result, logits = assess_skills(model, records, data)
    save_file({"logits": logits}, output / "logits.safetensors")
    write(
        output / "results.json",
        {
            "metrics": result,
            "logits_sha256": digest(output / "logits.safetensors"),
            "data_audit_sha256": digest(data / "audit.json"),
            "calibration_or_final_used": False,
        },
    )
    write(
        output / "complete.json",
        {
            "completed": True,
            "results_sha256": digest(output / "results.json"),
            "release_allowed": False,
        },
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
