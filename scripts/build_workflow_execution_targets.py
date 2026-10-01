"""Derive prospective rule-execution supervision from original train/development text only."""

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write
from verify_workspace_calibration_data import expression

from veyra.data import TrainingRecord
from veyra.workflow_facts import stated_probabilities

RECORDS = Path("data/workflow-v12/records.jsonl")
RECORDS_SHA = "145e01a41ed73f9c02524e5bf9a8a0efa8b73300eb9871b30ab0173b29ab5571"
OUTPUT = Path("data/workflow-v12-execution-targets")
ALLOWED = {"train", "dev"}
ACTIONS = ("PROCEED", "REVIEW", "HOLD")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entropy(probabilities):
    return -sum(value * math.log2(value) for value in probabilities if value > 0)


def derive(record):
    if record.split not in ALLOWED:
        raise ValueError("execution targets must not inspect calibration/final text")
    text = record.request.state.text
    probabilities = stated_probabilities(text)
    rules = re.findall(r"^(\d+)\. IF (.+) THEN (PROCEED|REVIEW|HOLD)\.$", text, re.MULTILINE)
    defaults = re.findall(r"^Otherwise (PROCEED|REVIEW|HOLD)\.$", text, re.MULTILINE)
    if not rules or len(defaults) != 1:
        raise ValueError("expected one explicit priority policy")
    if [int(number) for number, _, _ in rules] != list(range(1, len(rules) + 1)):
        raise ValueError("rule numbers do not follow their rendered order")
    actions = [action for _, _, action in rules] + defaults
    satisfaction, first_match = [0.0] * len(rules), [0.0] * len(actions)
    for assignment in itertools.product((False, True), repeat=4):
        facts = dict(zip("ABCD", assignment, strict=True))
        mass = math.prod(
            probability if value else 1 - probability
            for value, probability in zip(assignment, probabilities, strict=True)
        )
        matches = [bool(expression(policy, facts)) for _, policy, _ in rules]
        first_match[next((i for i, value in enumerate(matches) if value), len(rules))] += mass
        for i, value in enumerate(matches):
            satisfaction[i] += mass * value
    if not math.isclose(sum(first_match), 1.0, abs_tol=1e-12):
        raise ValueError("branch probabilities do not sum to one")
    if any(not 0 <= value <= 1 + 1e-12 for value in satisfaction + first_match):
        raise ValueError("invalid execution probability")
    outcomes = dict.fromkeys(ACTIONS, 0.0)
    for action, mass in zip(actions, first_match, strict=True):
        outcomes[action] += mass
    name, question = next(iter(record.request.questions.items()))
    if question.type == "noul":
        if question.instructions not in {
            "Under the supplied rules, is HOLD the required outcome?",
            "Under the supplied rules and stated uncertainty, what is the probability "
            "that HOLD is required?",
        }:
            raise ValueError("binary proposition is not the declared HOLD projection")
        target = {"false": 1 - outcomes["HOLD"], "true": outcomes["HOLD"]}
    else:
        criteria = (
            question.criteria
            if question.type == "choice"
            else {str(i): meaning for i, meaning in enumerate(question.criteria)}
        )
        target = {}
        for key, meaning in criteria.items():
            parsed = re.fullmatch(r"Outcome (PROCEED|REVIEW|HOLD)\.", meaning)
            if not parsed:
                raise ValueError("unexpected outcome description")
            target[key] = outcomes[parsed[1]]
    if set(target) != set(record.targets[name]):
        raise ValueError("projected candidate keys differ from original targets")
    error = max(abs(value - record.targets[name][key]) for key, value in target.items())
    if error > 1e-7:
        raise ValueError("branch projection differs from preserved original target: " + record.id)
    return {
        "id": record.id,
        "group_id": record.group_id,
        "split": record.split,
        "family": record.family,
        "state_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "fact_probabilities": probabilities,
        "rule_satisfaction_probabilities": satisfaction,
        "first_matching_branch_probabilities": first_match,
        "branch_outcomes": actions,
        "outcome_probabilities": outcomes,
        "projected_target_max_absolute_error": error,
        "branch_entropy_beyond_outcome_bits": entropy(first_match) - entropy(outcomes.values()),
        "domain": next(tag[7:] for tag in record.tags if tag.startswith("domain:")),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError("execution-target artifacts are immutable")
    if digest(RECORDS) != RECORDS_SHA:
        raise ValueError("original data changed")
    source_files = (
        Path(__file__),
        Path("scripts/verify_workspace_calibration_data.py"),
        Path("src/veyra/workflow_facts.py"),
    )
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_sha256": RECORDS_SHA,
        "source_sha256": {str(path): digest(path) for path in source_files},
        "scope": "Prospective training-target audit; no model fitting or inference.",
        "allowed_splits": sorted(ALLOWED),
        "method": (
            "Parse only eligible original train/development text; enumerate the 16 assignments "
            "under its declared independent priors and sensors. Record each rule's marginal "
            "truth and the mutually exclusive first matching rule (including default). Sum "
            "branches by their rendered action and check every original candidate target."
        ),
        "calibration_or_final_used": False,
        "current_study_changed": False,
        "runtime_solver_added": False,
        "limitations": (
            "Targets encode specified procedural assumptions, not real event frequencies. "
            "A loss on first-branch probabilities can duplicate outcome supervision when "
            "branch actions are unique. Extra annotation does not prove learnability, causal "
            "reasoning, or generalization. Training requires a separate prospective protocol."
        ),
    }
    write(OUTPUT / "protocol.json", protocol)
    items = []
    for line in RECORDS.open(encoding="utf-8"):
        raw = json.loads(line)
        # Filter before validating or interpreting any held-out request/target contents.
        if raw["split"] not in ALLOWED:
            continue
        if not {"domain:workflow_new", "domain:uncertainty"}.intersection(raw["tags"]):
            continue
        record = TrainingRecord.model_validate(raw)
        if record.request.state.images or len(record.request.questions) != 1:
            raise ValueError("unexpected prospective supervision population")
        items.append(derive(record))
    counts = Counter(item["split"] for item in items)
    if counts != {"train": 3840, "dev": 1200}:
        raise ValueError("unexpected eligible train/development counts")
    by_split, seen_states = defaultdict(list), {}
    for item in items:
        previous = seen_states.setdefault(item["state_sha256"], item)
        for key in (
            "group_id",
            "split",
            "fact_probabilities",
            "rule_satisfaction_probabilities",
            "first_matching_branch_probabilities",
            "branch_outcomes",
        ):
            if previous[key] != item[key]:
                raise ValueError("equivalent typed views have inconsistent execution annotations")
        by_split[item["split"]].append(item)
    groups = {split: {item["group_id"] for item in rows} for split, rows in by_split.items()}
    if groups["train"] & groups["dev"]:
        raise ValueError("supervision groups cross data roles")
    target_path = OUTPUT / "targets.jsonl"
    target_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in items),
        encoding="utf-8",
        newline="\n",
    )
    summary = {}
    for split, rows in by_split.items():
        distinct = {item["state_sha256"]: item for item in rows}.values()
        uncertain = [item for item in distinct if item["domain"] == "uncertainty"]
        summary[split] = {
            "questions": len(rows),
            "groups": len(groups[split]),
            "distinct_states": len(distinct),
            "families": sorted({item["family"] for item in rows}),
            "maximum_rules": max(len(item["rule_satisfaction_probabilities"]) for item in rows),
            "maximum_branches": max(len(item["branch_outcomes"]) for item in rows),
            "states_with_repeated_branch_outcomes": sum(
                len(set(item["branch_outcomes"])) < len(item["branch_outcomes"])
                for item in distinct
            ),
            "uncertain_states": len(uncertain),
            "uncertain_mean_branch_entropy_beyond_outcome_bits": sum(
                item["branch_entropy_beyond_outcome_bits"] for item in uncertain
            )
            / len(uncertain),
            "uncertain_states_with_extra_branch_entropy": sum(
                item["branch_entropy_beyond_outcome_bits"] > 1e-8 for item in uncertain
            ),
        }
    audit = {
        "passed": True,
        "protocol_sha256": digest(OUTPUT / "protocol.json"),
        "targets_sha256": digest(target_path),
        "summary": summary,
        "maximum_original_target_absolute_error": max(
            item["projected_target_max_absolute_error"] for item in items
        ),
        "no_training_development_group_overlap": True,
        "equivalent_view_annotations_match": True,
        "calibration_or_final_used": False,
        "model_training_or_inference_used": False,
        "release_allowed": False,
    }
    write(OUTPUT / "audit.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
