"""Fit one deployed threshold subject to separate empirical cohort constraints."""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from veyra.candidates import candidates_for
from veyra.data import read_records


@lru_cache(maxsize=1)
def cohort_groups():
    config = json.loads(
        Path("configs/workflow-recovery-policy-v13.json").read_text(encoding="utf-8")
    )
    groups = {}
    for name, population in config["fitting_populations"].items():
        path = Path(population["records"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != population["records_sha256"]:
            raise ValueError("Declared fitting population changed")
        for record in read_records(path):
            if record.split == "calibration":
                if record.group_id in groups and groups[record.group_id] != name:
                    raise ValueError("Calibration cohorts share an observation group")
                groups[record.group_id] = name
    return groups


def fit_policy(indices, logits, descriptors, temperature, max_error, minimum_coverage, device):
    config = json.loads(
        Path("configs/workflow-recovery-policy-v13.json").read_text(encoding="utf-8")
    )
    method = config["method"]
    if (
        method["name"] != "cohort_robust_empirical_policy"
        or max_error != 0.12
        or minimum_coverage != 0.6
        or method["threshold_grid_denominator"] != 200
    ):
        raise ValueError("Unknown cohort fitting rule")
    observations = {name: [] for name in config["fitting_populations"]}
    groups = cohort_groups()
    for index in indices:
        record, name, question = descriptors[index]
        probabilities = (logits[index].to(device).float() / temperature).softmax(-1)
        winner = candidates_for(question)[int(probabilities.argmax())].key
        gold = next((tag[9:] for tag in record.tags if tag.startswith("gold_key:")), None)
        error = (
            float(winner != gold)
            if "teacher_distribution" in record.tags and gold is not None
            else 1 - record.targets[name][winner]
        )
        observations[groups[record.group_id]].append((float(probabilities.max()), error))
    if any(len(rows) < 20 for rows in observations.values()):
        raise ValueError("A declared type/cohort has too few policy-fitting observations")
    # A fixed ascending grid maximizes coverage without adapting grid points to validation.
    for step in range(201):
        threshold = step / 200
        by_cohort = {}
        for cohort, rows in observations.items():
            accepted = [error for confidence, error in rows if confidence >= threshold]
            error = sum(accepted) / len(accepted) if accepted else None
            by_cohort[cohort] = {
                "questions": len(rows),
                "accepted": len(accepted),
                "coverage": len(accepted) / len(rows),
                "expected_error": error,
                "passed": len(accepted) >= max(20, math.ceil(len(rows) * minimum_coverage))
                and error is not None
                and error <= max_error,
            }
        if all(value["passed"] for value in by_cohort.values()):
            answers = sum(value["accepted"] for value in by_cohort.values())
            count = sum(value["questions"] for value in by_cohort.values())
            return {
                "threshold": threshold,
                "accepted": answers,
                "questions": count,
                "coverage": answers / count,
                "expected_error": sum(
                    value["accepted"] * value["expected_error"] for value in by_cohort.values()
                )
                / answers,
                "by_cohort": by_cohort,
                "always_abstain": False,
                "passed": True,
            }
    return {
        "threshold": 1.0,
        "accepted": 0,
        "questions": len(indices),
        "coverage": 0.0,
        "expected_error": None,
        "by_cohort": by_cohort,
        "always_abstain": True,
        "passed": False,
    }
