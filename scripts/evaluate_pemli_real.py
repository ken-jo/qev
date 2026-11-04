"""Run either checkpoint on the same untouched records; one model resident at a time."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from qev.pemli_real import RealDecisionModel
from veyra.candidates import candidates_for
from veyra.data import read_records


def evaluate(model, records, root, device):
    rows, latencies = [], []
    for record in records:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(record.request, image_root=root)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        for name, q in record.request.questions.items():
            answer = result["answers"][name]
            keys = [c.key for c in candidates_for(q)]
            probs = np.array([answer["probabilities"][k] for k in keys])
            target = np.array([record.targets[name][k] for k in keys])
            rows.append(
                {
                    "id": record.id,
                    "question": name,
                    "group": record.group_id,
                    "type": q.type,
                    "correct": float(target[int(probs.argmax())]),
                    "nll": float(-(target * np.log(np.maximum(probs, 1e-12))).sum()),
                    "brier": float(((probs - target) ** 2).sum()),
                    "confidence": float(probs.max()),
                    "abstained": answer["abstained"],
                    "score_mae": abs(answer["score"] - float(np.arange(len(keys)) @ target))
                    if q.type == "score"
                    else None,
                    "answer": answer,
                }
            )
    accepted = [r for r in rows if not r["abstained"]]
    ece = 0.0
    for i in range(10):
        bucket = [
            r
            for r in rows
            if i / 10 <= r["confidence"] and (r["confidence"] < (i + 1) / 10 or i == 9)
        ]
        if bucket:
            ece += (
                len(bucket)
                / len(rows)
                * abs(
                    np.mean([r["correct"] for r in bucket])
                    - np.mean([r["confidence"] for r in bucket])
                )
            )
    return {
        "rows": rows,
        "summary": {
            "questions": len(rows),
            "accuracy": float(np.mean([r["correct"] for r in rows])),
            "nll": float(np.mean([r["nll"] for r in rows])),
            "brier": float(np.mean([r["brier"] for r in rows])),
            "ece": float(ece),
            "coverage": len(accepted) / len(rows),
            "accepted_error": 1 - float(np.mean([r["correct"] for r in accepted]))
            if accepted
            else None,
            "score_mae": float(
                np.mean([r["score_mae"] for r in rows if r["score_mae"] is not None])
            )
            if any(r["score_mae"] is not None for r in rows)
            else None,
            "request_p50_ms": float(np.percentile(latencies, 50)),
            "request_p95_ms": float(np.percentile(latencies, 95)),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["pemli", "qev08"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("evaluation output is immutable")
    records = [r for r in read_records(a.records) if r.split == a.split]
    if not records:
        raise ValueError("empty evaluation split")
    if a.backend == "pemli":
        provenance = json.loads((a.checkpoint / "manifest.json").read_text())["provenance"]
        used_groups = set(provenance["training_groups"]) | set(
            provenance.get("calibration", {}).get("groups", [])
        )
        if {r.group_id for r in records} & used_groups:
            raise ValueError("evaluation groups overlap training")
        model = RealDecisionModel.load(a.checkpoint, a.device)
        model.config.cache_entries = 0
    else:
        from veyra.option_model import OptionModel

        model = OptionModel.load(a.checkpoint, device=a.device, local_files_only=True, merge=True)
        if "0.8b" not in model.backbone["model_id"].lower():
            raise ValueError("baseline checkpoint is not QEV-0.8B")
    # Model load and one warmup are excluded from resident latency, reported separately.
    model.predict(records[0].request, image_root=a.records.parent)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    result = evaluate(model, records, a.records.parent, a.device)
    result.update(
        {
            "backend": a.backend,
            "checkpoint": str(a.checkpoint.resolve()),
            "checkpoint_manifest_sha256": hashlib.sha256(
                (a.checkpoint / "manifest.json").read_bytes()
            ).hexdigest(),
            "records_sha256": hashlib.sha256(a.records.read_bytes()).hexdigest(),
            "subset_sha256": hashlib.sha256(
                "\n".join(r.model_dump_json() for r in records).encode()
            ).hexdigest(),
            "split": a.split,
            "device": a.device,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cache": "disabled",
            "timing": "resident serial, warmup excluded",
            "gpu": torch.cuda.get_device_name() if a.device.startswith("cuda") else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if a.device.startswith("cuda")
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()
            if a.device.startswith("cuda")
            else None,
        }
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
