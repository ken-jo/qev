"""Training-only interventions for controlled policies; never called by inference."""

from __future__ import annotations

import random
import re
from collections import defaultdict

from veyra.candidates import candidates_for
from veyra.data import TrainingRecord
from veyra.evidence_data import original_branch, policy_branches
from veyra.policy_data import typed_decision


def retype_unit(unit: list[TrainingRecord], rng: random.Random) -> list[TrainingRecord]:
    """Change the requested output type/proposition while preserving the observed rule branch."""
    if any(record.split != "train" for record in unit):
        raise ValueError("type interventions are restricted to training records")
    if len(unit) == 1:
        return list(unit)  # Primitive-condition supervision keeps its original question.
    if len(unit) != 2 or unit[0].group_id != unit[1].group_id:
        raise ValueError("expected a controlled counterfactual pair")
    ordered = sorted(unit, key=lambda record: record.id)
    anchor = ordered[0]
    tags = [tag for tag in anchor.tags if tag.startswith("oracle_condition_branch:")]
    branch = int(tags[0].split(":")[1]) if tags else original_branch(anchor)
    labels = list(dict.fromkeys(policy_branches(anchor)[1]))
    kind = rng.choice(("choice", "score", "noul"))
    probe = rng.choice(labels)
    rng.shuffle(labels)
    transformed = []
    for record in unit:
        name, question = next(iter(record.request.questions.items()))
        if len(record.request.questions) != 1:
            raise ValueError("controlled type interventions require one question")
        instructions = re.sub(r" Is the policy outcome '[^']+'\?$", "", question.instructions)
        instructions = instructions.removesuffix(
            " Select the ordered level whose description matches the policy outcome."
        )
        answer = policy_branches(record)[1][branch]
        converted, target = typed_decision(rng, instructions, labels, answer, kind, probe)
        raw = record.model_dump()
        raw["request"]["questions"] = {name: converted}
        raw["targets"] = {name: target}
        raw["tags"] = [tag for tag in raw["tags"] if not tag.startswith("oracle_condition_branch:")]
        raw["tags"] += [f"oracle_condition_branch:{branch}", "training_type_intervention"]
        transformed.append(TrainingRecord.model_validate(raw))
    return transformed


def primitive_question(record: TrainingRecord) -> TrainingRecord:
    """Expose the condition-reading task, without feeding its answer to the main model."""
    if record.split != "train":
        raise ValueError("auxiliary questions are restricted to training records")
    name, question = next(iter(record.request.questions.items()))
    branches = re.findall(r"If (.*?), return '([^']+)'\.", question.instructions)
    if not branches:
        raise ValueError("expected a controlled policy rule")
    candidates = candidates_for(question)
    winner = max(candidates, key=lambda c: record.targets[name][c.key])
    branch_tags = [tag for tag in record.tags if tag.startswith("oracle_condition_branch:")]
    if branch_tags:
        branch = int(branch_tags[0].split(":")[1])
        outcomes = re.findall(r"return '([^']+)'", question.instructions)
        answer = outcomes[branch]
    elif question.type == "noul":
        if winner.key != "true":
            raise ValueError("primitive extraction requires the first true-probe counterfactual")
        answer = re.search(r"Is the policy outcome '([^']+)'\?", question.instructions).group(1)
    else:
        answer = winner.description
    if len(branches) == 1:
        condition, outcome = branches[0]
        auxiliary = {
            "type": "noul",
            "instructions": f"Using the observation, is this condition true: {condition}?",
        }
        target = {"false": float(answer != outcome), "true": float(answer == outcome)}
    else:
        auxiliary = {
            "type": "choice",
            "instructions": "Which condition is satisfied by the observation?",
            "criteria": {str(i): condition for i, (condition, _) in enumerate(branches)},
        }
        target = {str(i): float(answer == outcome) for i, (_, outcome) in enumerate(branches)}
    raw = record.model_dump()
    raw.update(
        id=record.id + "-primitive",
        group_id=record.group_id + "-primitive",
        targets={"condition": target},
        tags=record.tags + ["training_auxiliary_condition"],
    )
    raw["request"]["questions"] = {"condition": auxiliary}
    return TrainingRecord.model_validate(raw)


def training_units(records: list[TrainingRecord], auxiliary=True) -> list[list[TrainingRecord]]:
    groups = defaultdict(list)
    for record in records:
        if record.split != "train":
            raise ValueError("intervention training cannot consume held-out splits")
        if len(record.request.questions) != 1:
            raise ValueError("controlled intervention records require one question")
        groups[record.group_id].append(record)
    units = []
    for group in groups.values():
        group.sort(key=lambda r: r.id)
        if len(group) != 2:
            raise ValueError("expected two counterfactuals per observation")
        units.append(group)
        if auxiliary:
            units.append([primitive_question(group[0])])
    return units


def augment_unit(unit: list[TrainingRecord], rng: random.Random) -> list[TrainingRecord]:
    """Rename symbols and shift text numbers jointly; all original targets are preserved."""
    if any(record.split != "train" for record in unit):
        raise ValueError("interventions cannot modify held-out inputs")
    # Rename only quoted output meanings, not generic true/false or primitive conditions.
    labels = sorted(
        {
            label
            for record in unit
            for question in record.request.questions.values()
            for label in re.findall(r"return '([^']+)'", question.instructions)
        }
    )
    aliases = {
        label: "route " + "".join(rng.sample("abcdefghijkmnpqrstuvwxyz", 6)) for label in labels
    }
    fields = {}
    if all(record.family.startswith("text_") for record in unit):
        for field in re.findall(r"(?:Observation: |; )([^=]+?) =", unit[0].request.state.text):
            fields[field] = "measure " + "".join(rng.sample("abcdefghijkmnpqrstuvwxyz", 4))
    replacement = {**aliases, **fields}
    pattern = (
        re.compile(
            r"(?<!\w)(?:"
            + "|".join(re.escape(s) for s in sorted(replacement, key=len, reverse=True))
            + r")(?!\w)"
        )
        if replacement
        else None
    )
    # A common nonnegative offset preserves every threshold/range/AND/OR truth value.
    offset = rng.choice((0, 0, 10, 20, 50, 100)) if fields else 0

    def rewrite(value):
        if pattern:
            value = pattern.sub(lambda match: replacement[match.group()], value)
        if offset:
            value = re.sub(r"\b\d+\b", lambda m: str(int(m.group()) + offset), value)
        return value

    augmented = []
    for record in unit:
        raw = record.model_dump()
        raw["request"]["state"]["text"] = rewrite(raw["request"]["state"]["text"])
        for question in raw["request"]["questions"].values():
            question["instructions"] = rewrite(question["instructions"])
            if question["type"] == "choice":
                question["criteria"] = {
                    key: rewrite(value) for key, value in question["criteria"].items()
                }
            elif question["type"] == "score":
                question["criteria"] = [rewrite(value) for value in question["criteria"]]
        augmented.append(TrainingRecord.model_validate(raw))
    return augmented
