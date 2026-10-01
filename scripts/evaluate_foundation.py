"""One immutable merged-model evaluation; optional calibration uses separate groups."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from train_foundation_head import write

from veyra.calibrate import fit_abstention, fit_temperature
from veyra.candidates import candidates_for
from veyra.constants import QUESTION_TYPES
from veyra.data import read_records
from veyra.decision_metrics import observation, report
from veyra.features import FeatureSample
from veyra.option_model import OptionModel
from veyra.probability import Calibration, typed_answer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "calibration", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrated-output", type=Path)
    parser.add_argument("--legacy-regression", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("evaluation output must be empty")
    if args.calibrated_output and args.split != "calibration":
        raise ValueError("only the calibration split can fit calibration")
    records = [r for r in read_records(args.records) if r.split == args.split]
    if not records:
        raise ValueError("empty evaluation")
    protocol = {
        "arguments": vars(args),
        "weights_sha256": digest(args.checkpoint / "head.safetensors"),
        "manifest_sha256": digest(args.checkpoint / "manifest.json"),
        "records_sha256": digest(args.records),
        "source_sha256": digest(Path(__file__)),
        "dependency_source_sha256": {
            str(path): digest(path)
            for path in (
                Path("src/veyra/decision_metrics.py"),
                Path("src/veyra/probability.py"),
                Path("src/veyra/option_model.py"),
                Path("src/veyra/calibrate.py"),
            )
        },
        "subset_sha256": hashlib.sha256(
            "".join(r.model_dump_json() + "\n" for r in records).encode()
        ).hexdigest(),
        "metric": (
            "All probability-argmax decisions, including abstained answers, "
            "against explicit hard gold; soft targets scored separately"
        ),
        "merged_bf16_deployment": True,
        "probability_serialization": "typed_answer float32 softmax on the model device",
    }
    write(args.output / "protocol.json", protocol)
    model = OptionModel.load(args.checkpoint, local_files_only=True, merge=True)
    torch.set_num_threads(4)
    samples, logits, descriptors = [], [], []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        output, _ = model(record.request, args.records.parent)
        for name, value in output.items():
            question = record.request.questions[name]
            candidates = candidates_for(question)
            samples.append(
                FeatureSample(
                    features=torch.zeros(len(candidates), 1),
                    context=torch.zeros(1),
                    levels=torch.tensor([c.level for c in candidates]),
                    targets=torch.tensor([record.targets[name][c.key] for c in candidates]),
                    type_id=QUESTION_TYPES.index(question.type),
                    record_id=record.id,
                    group_id=record.group_id,
                    split=record.split,
                    family=record.family,
                    language=record.language,
                    question_id=name,
                    tags=record.tags,
                )
            )
            logits.append(value.cpu().float())
            descriptors.append((record, name, question))
        if index % 200 == 0:
            print(
                json.dumps({"evaluated": index, "total": len(records), "split": args.split}),
                flush=True,
            )
    calibration = model.calibration
    fitting_report = None
    if args.calibrated_output:
        temperatures, thresholds, always, fitting_report = [], [], [], {}
        for type_id, kind in enumerate(QUESTION_TYPES):
            fitting, policy = [], []
            for i, sample in enumerate(samples):
                if sample.type_id == type_id:
                    bucket = int(hashlib.sha256(sample.group_id.encode()).hexdigest()[:8], 16) % 2
                    (fitting if bucket == 0 else policy).append(i)
            temperature = max(
                1.0, fit_temperature([samples[i] for i in fitting], [logits[i] for i in fitting])
            )
            selection = fit_abstention(
                [samples[i] for i in policy], [logits[i] for i in policy], temperature
            )
            temperatures.append(temperature)
            thresholds.append(selection["threshold"])
            if selection["always_abstain"]:
                always.append(kind)
            fitting_report[kind] = {
                **selection,
                "temperature": temperature,
                "temperature_questions": len(fitting),
                "policy_questions": len(policy),
            }
        calibration = Calibration(
            tuple(temperatures),
            tuple(thresholds),
            tuple(QUESTION_TYPES),
            tuple(QUESTION_TYPES),
            tuple(always),
        )
        if args.calibrated_output.exists() and any(args.calibrated_output.iterdir()):
            raise FileExistsError("calibrated checkpoint output must be empty")
        args.calibrated_output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            args.checkpoint / "head.safetensors", args.calibrated_output / "head.safetensors"
        )
        manifest = json.loads((args.checkpoint / "manifest.json").read_text())
        manifest["calibration"] = calibration.to_dict()
        manifest["training"]["intermediate"] = False
        manifest["training"]["calibration"] = {
            "split": "calibration",
            "temperature_floor": 1.0,
            "group_disjoint": True,
            "protocol": protocol,
            "fits": fitting_report,
        }
        write(args.calibrated_output / "manifest.json", manifest)
    rows = []
    for (record, name, question), value in zip(descriptors, logits, strict=True):
        answer = typed_answer(question, value.to(model.encoder.device), calibration)
        row = observation(record, name, answer["probabilities"], abstained=answer["abstained"])
        row["confidence"] = answer["confidence"]
        if args.legacy_regression:
            row["scope"] = "legacy_regression"
        rows.append(row)
    prediction_path = args.output / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "protocol": protocol,
        **report(rows, bootstrap=args.split == "test"),
        "calibration": calibration.to_dict(),
        "fitting": fitting_report,
        "predictions_sha256": digest(prediction_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write(args.output / "evaluation.json", result)
    print(json.dumps({"overall": result["overall"], "by_domain": result["by_domain"]}), flush=True)


if __name__ == "__main__":
    main()
