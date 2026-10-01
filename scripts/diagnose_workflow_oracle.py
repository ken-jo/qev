"""Development-only reference limits under the declared annotation/evidence model."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from train_foundation_head import write

from veyra.constants import QUESTION_TYPES
from veyra.data import read_records


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    # Knowing the annotation is an ideal reference, never a model input or prediction.
    ordered = sorted(rows, key=lambda row: (row["minimum_expected_error"], row["id"]))
    total = 0.0
    maximum = 0
    for count, row in enumerate(ordered, 1):
        total += row["minimum_expected_error"]
        boundary = count == len(ordered) or (
            ordered[count]["minimum_expected_error"] != row["minimum_expected_error"]
        )
        if boundary and count >= 20 and total / count <= 0.15:
            maximum = count
    return {
        "questions": len(rows),
        "groups": len({row["group"] for row in rows}),
        "hard_annotation_questions": sum(row["hard_annotation"] for row in rows),
        "known_conditional_questions": sum(row["known_conditional"] for row in rows),
        "maximum_ideal_coverage_at_15pct_error": maximum / len(rows),
        "ideal_risk_at_coverage": {
            str(coverage): sum(
                row["minimum_expected_error"]
                for row in ordered[: max(1, math.floor(len(rows) * coverage))]
            )
            / max(1, math.floor(len(rows) * coverage))
            for coverage in (0.5, 0.6, 0.8, 1.0)
        },
    }


def main():
    output = Path("runs/workflow-v12-development-oracle.json")
    if output.exists():
        raise FileExistsError("development reference is immutable")
    source = Path("data/workflow-v12/records.jsonl")
    rows = []
    for record in read_records(source):
        if record.split != "dev":
            continue
        gold = next((tag[9:] for tag in record.tags if tag.startswith("gold_key:")), None)
        domain = next(tag[7:] for tag in record.tags if tag.startswith("domain:"))
        condition = next((tag[10:] for tag in record.tags if tag.startswith("condition:")), "none")
        for name, question in record.request.questions.items():
            target = record.targets[name]
            if abs(sum(target.values()) - 1) > 1e-6:
                raise ValueError("target probabilities are not normalized")
            hard = gold is not None or max(target.values()) >= 1 - 1e-6
            # Official teacher-distribution rows are scored against their recorded hard gold.
            oracle_error = 0.0 if gold is not None else 1 - max(target.values())
            rows.append(
                {
                    "id": record.id,
                    "group": record.group_id,
                    "domain": domain,
                    "condition": condition,
                    "type": question.type,
                    "hard_annotation": hard,
                    "known_conditional": "known_conditional_distribution" in record.tags,
                    "minimum_expected_error": oracle_error,
                }
            )
    if set(row["type"] for row in rows) != set(QUESTION_TYPES):
        raise ValueError("incomplete development type coverage")
    result = {
        "scope": "Development annotation-aware ideal reference; not achieved model performance",
        "limitations": (
            "Hard-label agreement is assumed perfect, including teacher labels. Soft targets "
            "follow the stated conditional-independence and sensor/prior assumptions. This "
            "neither proves these targets describe real business frequencies nor fits a "
            "deployable policy. Fixed-count risk points can break ideal-confidence ties; "
            "maximum coverage respects complete confidence ties."
        ),
        "records_sha256": digest(source),
        "source_sha256": digest(Path(__file__)),
        "calibration_or_final_used": False,
        "deployed_policy_changed": False,
        "overall": summarize(rows),
    }
    for field in ("type", "domain", "condition"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result["by_" + field] = {key: summarize(value) for key, value in sorted(groups.items())}
    write(output, result)
    print(
        json.dumps(
            {"by_type": result["by_type"], "uncertainty": result["by_domain"]["uncertainty"]}
        )
    )


if __name__ == "__main__":
    main()
