"""Separate oracle risk and model regret on the already consumed v12 final only."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("runs/workflow-v12-cohort-policy-release")
DATA = Path("data/workflow-v12-cohort/records.jsonl")
OUTPUT = Path("runs/backbone-recovery-diagnostics-v13/consumed-final-uncertainty.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_cost(target, action, critical):
    return 1 - target[action] + (4 * target[critical] if action != critical else 0)


def summarize(rows, population_size):
    if not rows:
        return {"questions": 0, "coverage": 0}
    fields = (
        "expected_error",
        "minimum_error",
        "decision_regret",
        "expected_cost",
        "minimum_cost",
        "cost_regret",
        "nll",
        "entropy",
        "nll_excess",
        "brier",
    )
    return {
        "questions": len(rows),
        "groups": len({r["group"] for r in rows}),
        "coverage": len(rows) / population_size,
        **{key: math.fsum(r[key] for r in rows) / len(rows) for key in fields},
    }


def main():
    require(not OUTPUT.exists(), "diagnosis is immutable")
    require(
        digest(DATA) == "c22d9140b3f934de4bb41fe65d73ef6180553202c39bd1e874fd545b2c1c82d2",
        "previously evaluated data changed",
    )
    records = {
        r["id"]: r
        for line in DATA.read_text(encoding="utf-8").splitlines()
        if (r := json.loads(line))["split"] == "test"
    }
    design = Path("configs/workflow-release-v12.json")
    evidence = {str(DATA): digest(DATA), str(design): digest(design)}
    populations, identities = {}, {}
    weights = {
        "baseline": "f13aa3346d7da369a2769eef7fa487747c97d1ba0878ea3af8f713463f463491",
        "final": "86ce942f8ef036f2279422a3a2665248b98adcb0effa91bafdf15d9c14c6c83e",
    }
    for name in ("baseline", "final"):
        report_path, prediction_path = (
            ROOT / name / "evaluation.json",
            ROOT / name / "predictions.jsonl",
        )
        report = read(report_path)
        require(
            report["protocol"]["records_sha256"] == digest(DATA)
            and report["protocol"]["weights_sha256"] == weights[name]
            and report["predictions_sha256"] == digest(prediction_path),
            "historical model or prediction identity changed",
        )
        rows = [
            r
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if (r := json.loads(line))["domain"] == "uncertainty"
        ]
        require(len(rows) == len({r["id"] for r in rows}) == 480, "wrong uncertainty population")
        require(
            len({r["group"] for r in rows}) == 480
            and all(
                sum(r["condition"] == name for r in rows) == 160
                for name in ("missing", "conflicting", "shifted_prior")
            ),
            "uncertainty group or condition counts changed",
        )
        for row in rows:
            record = records[row["id"]]
            target = record["targets"][row["question"]]
            critical = next(t[13:] for t in record["tags"] if t.startswith("critical_key:"))
            require(
                row["targets"] == target
                and row["critical_key"] == critical
                and row["group"] == record["group_id"]
                and "known_conditional_distribution" in record["tags"],
                "conditional target or identity changed",
            )
            p, action = row["probabilities"], row["prediction"]
            require(
                set(p) == set(target) and abs(math.fsum(p.values()) - 1) < 1e-6,
                "invalid probability support",
            )
            entropy = -math.fsum(v * math.log(v) for v in target.values() if v > 0)
            nll = -math.fsum(target[k] * math.log(max(1e-12, p[k])) for k in p)
            brier = math.fsum((p[k] - target[k]) ** 2 for k in p)
            error, cost = 1 - target[action], expected_cost(target, action, critical)
            for computed, key in (
                (nll, "nll"),
                (brier, "brier"),
                (error, "expected_error"),
                (cost, "expected_cost"),
            ):
                require(abs(computed - row[key]) <= 1e-9, "saved metric does not reconstruct")
            minimum_error = 1 - max(target.values())
            minimum_cost = min(expected_cost(target, key, critical) for key in target)
            row.update(
                minimum_error=minimum_error,
                decision_regret=error - minimum_error,
                minimum_cost=minimum_cost,
                cost_regret=cost - minimum_cost,
                entropy=entropy,
                nll_excess=nll - entropy,
            )
            require(
                min(row["decision_regret"], row["cost_regret"], row["nll_excess"]) >= -1e-9,
                "oracle decomposition produced negative excess risk",
            )
        populations[name] = rows
        identities[name] = {r["id"]: (r["group"], r["targets"]) for r in rows}
        for path in (report_path, prediction_path):
            evidence[str(path)] = digest(path)
    require(identities["baseline"] == identities["final"], "models used different observations")
    common_accepted = {r["id"] for r in populations["final"] if not r["abstained"]}
    summaries = {}
    for name, rows in populations.items():
        ordered = sorted(
            rows, key=lambda r: (-r["confidence"], hashlib.sha256(r["id"].encode()).hexdigest())
        )
        summaries[name] = {
            "all_requests": summarize(rows, 480),
            "own_policy_accepted": summarize([r for r in rows if not r["abstained"]], 480),
            "same_cohort_policy_accepted_requests": summarize(
                [r for r in rows if r["id"] in common_accepted], 480
            ),
            "own_confidence_top_80_percent": summarize(ordered[:384], 480),
            "all_requests_by_condition": {
                condition: summarize([r for r in rows if r["condition"] == condition], 160)
                for condition in ("missing", "conflicting", "shifted_prior")
            },
        }
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(__file__),
        "evidence_files": evidence,
        "scope": "Post-hoc diagnosis of the consumed v12 final; not a recovery-model result.",
        "new_model_inference": False,
        "fresh_v13_data_used": False,
        "used_for_training_or_candidate_selection": False,
        "definitions": {
            "minimum_error": "1 - max of the exact conditional target distribution.",
            "minimum_cost": "Smallest conditional 0/1/5 loss over available actions.",
            "decision_regret": "Model expected error minus oracle error on identical requests.",
            "nll_excess": "NLL minus conditional entropy; KL up to the scoring epsilon.",
            "brier": "Squared error against the conditional distribution; oracle minimum is zero.",
        },
        "results": summaries,
        "limitations": [
            "Oracle probabilities come from synthetic targets and are not available to the model.",
            "An error-minimizing action and a cost-minimizing action can differ.",
            "Bounds depend on the authored latent/sensor model, with no real-world guarantee.",
            "Own accepted sets differ; common-set summaries use the cohort model's fixed set.",
            "No policy, checkpoint ranking or original release threshold was changed.",
        ],
        "release_allowed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
