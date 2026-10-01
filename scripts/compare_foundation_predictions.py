"""Report paired group bootstrap differences without making a checkpoint selection."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    indexed = {(r["id"], r["question"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate prediction identity")
    report = json.loads(path.with_name("evaluation.json").read_text(encoding="utf-8"))
    if digest(path) != report["predictions_sha256"]:
        raise ValueError("predictions differ from the evaluation report")
    return indexed


def summarize(pairs):
    result = {"questions": len(pairs), "groups": len({a["group"] for a, _ in pairs})}
    for metric in ("correct", "nll", "brier"):
        grouped = defaultdict(list)
        for old, new in pairs:
            if old[metric] is not None and new[metric] is not None:
                grouped[old["group"]].append(float(new[metric]) - float(old[metric]))
        if not grouped:
            continue
        totals = np.array([[sum(v), len(v)] for v in grouped.values()])
        rng = np.random.default_rng(20260930)
        sampled = totals[rng.integers(len(totals), size=(2000, len(totals)))].sum(1)
        result["accuracy" if metric == "correct" else metric] = {
            "new_minus_reference": float(totals[:, 0].sum() / totals[:, 1].sum()),
            "paired_group_bootstrap_95_ci": np.quantile(
                sampled[:, 0] / sampled[:, 1], [0.025, 0.975]
            ).tolist(),
            "questions": int(totals[:, 1].sum()),
            "groups": len(totals),
        }
    return result


def compare(reference, new, text_only=False):
    current = {key: row for key, row in new.items() if not text_only or row["modality"] == "text"}
    if reference.keys() != current.keys():
        raise ValueError("comparison has unmatched observations")
    pairs = []
    for key in sorted(reference):
        a, b = reference[key], current[key]
        for field in ("group", "domain", "view", "modality", "type", "gold", "targets"):
            if a[field] != b[field]:
                raise ValueError(f"different {field} for {key}")
        pairs.append((a, b))
    result = {"overall": summarize(pairs)}
    for field in ("domain", "type", "modality", "scope"):
        result["by_" + field] = {
            value: summarize([(a, b) for a, b in pairs if a[field] == value])
            for value in sorted({a[field] for a, _ in pairs})
        }
    result["by_domain_view"] = {
        domain + "/" + view: summarize(
            [(a, b) for a, b in pairs if a["domain"] == domain and a["view"] == view]
        )
        for domain, view in sorted({(a["domain"], a["view"]) for a, _ in pairs})
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-root", type=Path, default=Path("runs/foundation-v11-evaluation")
    )
    parser.add_argument("--laya-root", type=Path, default=Path("runs/foundation-v11-laya"))
    args = parser.parse_args()
    output = args.evaluation_root / "paired-comparison.json"
    if output.exists():
        raise FileExistsError("paired reports are immutable")
    files = {
        "veyra": args.evaluation_root / "foundation-final/predictions.jsonl",
        "dynamic_v2": args.evaluation_root / "baseline-final/predictions.jsonl",
        "laya_base": args.laya_root / "foundation-base/predictions.jsonl",
        "laya_specialist": args.laya_root / "foundation-specialist/predictions.jsonl",
    }
    rows = {name: predictions(path) for name, path in files.items()}
    result = {
        "protocol": {
            "source_sha256": digest(Path(__file__)),
            "predictions_sha256": {name: digest(path) for name, path in files.items()},
            "direction": "Veyra Foundation minus the named reference",
            "bootstrap": "2000 resamples of paired observation groups, seed 20260930",
            "selection_used": False,
            "limitation": (
                "Descriptive marginal intervals without multiple-comparison correction; "
                "Veyra was adapted on these task families, LAYA checkpoints were not."
            ),
        },
        "vs_dynamic_v2": compare(rows["dynamic_v2"], rows["veyra"]),
        "vs_laya_base_text": compare(rows["laya_base"], rows["veyra"], text_only=True),
        "vs_laya_specialist_text": compare(rows["laya_specialist"], rows["veyra"], text_only=True),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value["overall"] for key, value in result.items() if key != "protocol"}))


if __name__ == "__main__":
    main()
