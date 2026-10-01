"""Paired observation-group comparisons for the frozen workflow final evaluation."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from train_foundation_head import write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("paired final report is immutable")
    left, right = load(args.before), load(args.after)
    old = {row["id"]: row for row in left}
    new = {row["id"]: row for row in right}
    if set(old) != set(new) or len(old) != len(left) or len(new) != len(right):
        raise ValueError("final observations do not align")
    for key in old:
        for field in ("group", "domain", "family", "condition", "gold", "targets"):
            if old[key][field] != new[key][field]:
                raise ValueError("paired input/label mismatch: " + field)
    rng = np.random.default_rng(20260930131)

    def paired(subset, field, hard=False):
        groups = defaultdict(list)
        for key in subset:
            if not hard or old[key]["correct"] is not None:
                groups[old[key]["group"]].append(key)
        totals = np.array(
            [
                [
                    sum(float(old[k][field]) for k in keys),
                    sum(float(new[k][field]) for k in keys),
                    len(keys),
                ]
                for keys in groups.values()
            ]
        )
        sums = totals.sum(0)
        boot = totals[rng.integers(len(totals), size=(2000, len(totals)))].sum(1)
        differences = (boot[:, 1] - boot[:, 0]) / boot[:, 2]
        return {
            "before": float(sums[0] / sums[2]),
            "after": float(sums[1] / sums[2]),
            "delta": float((sums[1] - sums[0]) / sums[2]),
            "delta_95_ci": np.quantile(differences, [0.025, 0.975]).tolist(),
            "groups": len(groups),
            "questions": int(sums[2]),
        }

    def matched_risk(subset):
        # Exactly one uncertain view per group. Keep bootstrap resampling paired.
        if len({old[k]["group"] for k in subset}) != len(subset):
            raise ValueError("uncertainty bootstrap requires one view per group")
        indices = np.arange(len(subset))
        confidence = [np.array([side[k]["confidence"] for k in subset]) for side in (old, new)]
        costs = [np.array([side[k]["expected_cost"] for k in subset]) for side in (old, new)]
        count = max(1, math.floor(len(indices) * 0.8))

        def cost(side, sampled):
            order = np.argsort(-confidence[side][sampled], kind="stable")[:count]
            return costs[side][sampled[order]].mean()

        before, after = cost(0, indices), cost(1, indices)
        samples = rng.integers(len(indices), size=(2000, len(indices)))
        delta = np.array([cost(1, sample) - cost(0, sample) for sample in samples])
        return {
            "coverage": 0.8,
            "accepted": count,
            "before": float(before),
            "after": float(after),
            "delta": float(after - before),
            "delta_95_ci": np.quantile(delta, [0.025, 0.975]).tolist(),
        }

    # Hash order breaks exact-confidence ties independently of source labels or split position.
    keys = sorted(old, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    workflow = [key for key in keys if old[key]["domain"] == "workflow_new"]
    uncertainty = [key for key in keys if old[key]["domain"] == "uncertainty"]
    result = {
        "before_sha256": digest(args.before),
        "after_sha256": digest(args.after),
        "source_sha256": digest(Path(__file__)),
        "bootstrap": {
            "seed": 20260930131,
            "samples": 2000,
            "unit": "observation_group",
            "multiple_comparison_adjusted": False,
        },
        "new_workflow": {
            "accuracy": paired(workflow, "correct", hard=True),
            "families": {
                family: paired(
                    [k for k in workflow if old[k]["family"] == family], "correct", hard=True
                )
                for family in sorted({old[k]["family"] for k in workflow})
            },
        },
        "uncertainty": {
            "nll": paired(uncertainty, "nll"),
            "brier": paired(uncertainty, "brier"),
            "matched_coverage_cost": matched_risk(uncertainty),
            "condition_groups": dict(Counter(old[k]["condition"] for k in uncertainty)),
            "conditions": {
                condition: {
                    "nll": paired(
                        [k for k in uncertainty if old[k]["condition"] == condition], "nll"
                    ),
                    "brier": paired(
                        [k for k in uncertainty if old[k]["condition"] == condition], "brier"
                    ),
                    "matched_coverage_cost": matched_risk(
                        [k for k in uncertainty if old[k]["condition"] == condition]
                    ),
                }
                for condition in sorted({old[k]["condition"] for k in uncertainty})
            },
        },
        "retention": {
            domain: paired([k for k in keys if old[k]["domain"] == domain], "correct", hard=True)
            for domain in ("photo_guard", "text_nli", "text_intent")
        },
    }
    write(args.output, result)
    print(
        json.dumps(
            {
                "new_workflow": result["new_workflow"]["accuracy"],
                "uncertainty": {
                    k: result["uncertainty"][k] for k in ("nll", "brier", "matched_coverage_cost")
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
