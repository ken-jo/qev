"""Actual-model policy evaluation with explicit calibration and untouched final splits."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from veyra.calibrate import fit_abstention, fit_temperature
from veyra.candidates import candidates_for
from veyra.constants import QUESTION_TYPES
from veyra.data import read_records
from veyra.features import FeatureSample
from veyra.option_model import OptionModel
from veyra.probability import Calibration
from veyra.training import summarize_predictions


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "calibration", "test"], default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrated-output", type=Path)
    parser.add_argument("--expected-subset-sha256", help="Reject a changed frozen evaluation set")
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Previously inspected benchmark; not final evidence",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evaluation reports are immutable; use a new output path")
    if args.calibrated_output and args.split != "calibration":
        raise ValueError("calibration can only use the calibration split")
    records = [r for r in read_records(args.records) if r.split == args.split]
    if not records:
        raise ValueError("the requested evaluation split is empty")
    subset_sha256 = hashlib.sha256(
        "".join(r.model_dump_json() + "\n" for r in records).encode()
    ).hexdigest()
    if args.expected_subset_sha256 and subset_sha256 != args.expected_subset_sha256:
        raise ValueError("evaluation subset does not match its frozen checksum")
    # Fit and evaluate on the same merged BF16 deployment path.
    model = OptionModel.load(args.checkpoint, local_files_only=True, merge=True)
    samples, logits = [], []
    for i, record in enumerate(records):
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
        if (i + 1) % 100 == 0:
            print(f"evaluated {i + 1}/{len(records)} {args.split} records", flush=True)
    calibration = model.calibration
    fitting_report = None
    if args.calibrated_output:
        temperatures, thresholds, always, fitting_report = [], [], [], {}
        for type_id, kind in enumerate(QUESTION_TYPES):
            fitting, policy = [], []
            for i, sample in enumerate(samples):
                if sample.type_id == type_id:
                    group_bucket = (
                        int(hashlib.sha256(sample.group_id.encode()).hexdigest()[:8], 16) % 2
                    )
                    (fitting if group_bucket == 0 else policy).append(i)
            # The previous release sharpened probabilities and failed to transfer. Do not sharpen.
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
        model.calibration = calibration
        manifest = json.loads((args.checkpoint / "manifest.json").read_text())
        training = {
            **manifest["training"],
            "calibration": {
                "split": "calibration",
                "temperature_floor": 1.0,
                "group_disjoint": True,
                "dataset_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
                "subset_sha256": subset_sha256,
                "fits": fitting_report,
            },
        }
        if args.calibrated_output.exists() and any(args.calibrated_output.iterdir()):
            raise FileExistsError("calibrated checkpoint output must be empty")
        args.calibrated_output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            args.checkpoint / "head.safetensors", args.calibrated_output / "head.safetensors"
        )
        manifest["training"] = training
        manifest["calibration"] = calibration.to_dict()
        (args.calibrated_output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

    def metrics(indices):
        selected = [samples[i] for i in indices]
        values = [logits[i] for i in indices]
        summary = summarize_predictions(selected, values, calibration.temperatures)
        groups, group_tags, accepted = {}, {}, []
        for sample, prediction in zip(selected, values, strict=True):
            probs = (prediction / calibration.temperatures[sample.type_id]).softmax(-1)
            correct = float(sample.targets[int(probs.argmax())])
            groups.setdefault(sample.group_id, []).append(correct)
            group_tags.setdefault(sample.group_id, set()).update(sample.tags)
            if (
                QUESTION_TYPES[sample.type_id] not in calibration.always_abstain_types
                and float(probs.max()) >= calibration.abstain_thresholds[sample.type_id]
            ):
                accepted.append(correct)
        summary["coverage"] = len(accepted) / len(selected)
        summary["accepted_expected_accuracy"] = sum(accepted) / len(accepted) if accepted else None
        pairs = [
            outcomes
            for name, outcomes in groups.items()
            if len(outcomes) == 2
            and {"rule_version_0", "rule_version_1"}.issubset(group_tags[name])
        ]
        summary["counterfactual_pairs"] = len(pairs)
        summary["both_correct_accuracy"] = (
            sum(all(x == 1 for x in pair) for pair in pairs) / len(pairs) if pairs else None
        )
        # Counterfactual requests are correlated: resample observation groups, not rows.
        group_values = list(groups.values())
        rng = np.random.default_rng(91)
        group_sums = np.array([sum(group) for group in group_values])
        group_sizes = np.array([len(group) for group in group_values])
        draws = rng.integers(0, len(groups), size=(2000, len(groups)))
        bootstrap = group_sums[draws].sum(1) / group_sizes[draws].sum(1)
        summary["group_bootstrap_expected_accuracy_95_ci"] = np.percentile(
            bootstrap, [2.5, 97.5]
        ).tolist()
        return summary

    report = {
        "split": args.split,
        "evaluation_role": "regression" if args.regression else args.split,
        "architecture": "option_readout",
        "accuracy_target": 0.9,
        "accuracy_target_scope": "pre-abstention, each modality separately",
        "calibration": calibration.to_dict(),
        "dataset_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "evaluated_subset_sha256": subset_sha256,
        "evaluated_records": len(records),
        "checkpoint_sha256": hashlib.sha256(
            (args.checkpoint / "head.safetensors").read_bytes()
        ).hexdigest(),
        "checkpoint_manifest_sha256": hashlib.sha256(
            (args.checkpoint / "manifest.json").read_bytes()
        ).hexdigest(),
        "overall": metrics(list(range(len(samples)))),
        "raw_overall": summarize_predictions(samples, logits),
        "by_modality": {
            kind: metrics([i for i, s in enumerate(samples) if s.family.startswith(kind + "_")])
            for kind in ("text", "image")
            if any(s.family.startswith(kind + "_") for s in samples)
        },
        "by_family": {
            kind: metrics([i for i, s in enumerate(samples) if s.family == kind])
            for kind in sorted({s.family for s in samples})
        },
        "by_type": {
            kind: metrics([i for i, s in enumerate(samples) if s.type_id == type_id])
            for type_id, kind in enumerate(QUESTION_TYPES)
            if any(s.type_id == type_id for s in samples)
        },
        "by_target_kind": {
            kind: metrics(indices)
            for kind in ("hard", "soft")
            if (
                indices := [
                    i
                    for i, sample in enumerate(samples)
                    if ("hard" if float(sample.targets.max()) >= 1 - 1e-6 else "soft") == kind
                ]
            )
        },
        "by_language": {
            kind: metrics([i for i, s in enumerate(samples) if s.language == kind])
            for kind in sorted({s.language for s in samples})
        },
        "by_tag": {
            tag: metrics([i for i, s in enumerate(samples) if tag in s.tags])
            for tag in (
                "numeric_boundary",
                "image_text_conflict",
                "fresh_policy",
                "fresh_diagram",
                "legacy_regression",
                "unseen_template",
                "unseen_color_allowed",
                "missing_evidence",
            )
            if any(tag in s.tags for s in samples)
        },
        "fitting": fitting_report,
        "calibrated_checkpoint_manifest_sha256": (
            hashlib.sha256((args.calibrated_output / "manifest.json").read_bytes()).hexdigest()
            if args.calibrated_output
            else None
        ),
        "predictions": [
            {
                "id": s.record_id,
                "question": s.question_id,
                "group": s.group_id,
                "logits": p.tolist(),
                "targets": s.targets.tolist(),
            }
            for s, p in zip(samples, logits, strict=True)
        ],
    }
    report["macro_family_expected_accuracy"] = sum(
        value["expected_accuracy"] for value in report["by_family"].values()
    ) / len(report["by_family"])
    if set(report["by_modality"]) == {"text", "image"}:
        report["accuracy_target_met"] = all(
            m["hard_label_accuracy"] >= report["accuracy_target"]
            for m in report["by_modality"].values()
        )
    report["final_accuracy_target_met"] = (
        args.split == "test" and not args.regression and report.get("accuracy_target_met", False)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "by_modality": report["by_modality"],
                "by_family": report["by_family"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
