"""Recompute skill labels from rendered propositions, separate from the data builder."""

import hashlib
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

from train_foundation_head import write
from verify_workspace_calibration_data import expression
from workflow_skill_common import canonical_target

from veyra.candidates import candidates_for
from veyra.data import read_records
from veyra.workflow_facts import stated_probabilities


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("data/workflow-v12-skills")
    output = root / "independent-target-audit.json"
    if output.exists():
        raise FileExistsError("skill target audit is immutable")
    counts = Counter()
    maximum_error = 0.0
    sources = {}
    for line in Path("data/workflow-v12/records.jsonl").open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] in {"train", "dev"}:
            text_hash = hashlib.sha256(raw["request"]["state"]["text"].encode()).hexdigest()
            sources.setdefault(text_hash, (raw["split"], raw["group_id"], raw["family"]))
    for split in ("train", "dev"):
        for record in read_records(root / f"{split}.jsonl"):
            state = record.request.state.text
            state_hash = hashlib.sha256(state.encode()).hexdigest()
            if sources[state_hash] != (record.split, record.group_id, record.family):
                raise ValueError("skill state crossed its original source role")
            name, question = next(iter(record.request.questions.items()))
            probabilities = stated_probabilities(state)
            fact = re.match(
                r"Proposition: Condition ([ABCD]), as defined in the evidence, is (true|false)\.",
                question.instructions,
            )
            rule = re.match(
                r"Proposition: The Boolean condition of rule (\d+) is (true|false)\.",
                question.instructions,
            )
            if fact:
                p, truth = probabilities["ABCD".index(fact[1])], fact[2]
                kind = "fact"
            elif rule:
                rules = re.findall(r"^\d+\. IF (.+) THEN (?:PROCEED|REVIEW|HOLD)\.$", state, re.M)
                predicate = rules[int(rule[1]) - 1]
                truth, p, kind = rule[2], 0.0, "rule"
                for values in itertools.product((False, True), repeat=4):
                    mass = math.prod(
                        probability if value else 1 - probability
                        for value, probability in zip(values, probabilities, strict=True)
                    )
                    p += mass * bool(expression(predicate, dict(zip("ABCD", values, strict=True))))
            else:
                raise ValueError("unexpected skill proposition")
            if truth == "false":
                p = 1 - p
            candidates = candidates_for(question)
            expected = {}
            for candidate in candidates:
                if candidate.description == "The proposition is true.":
                    expected[candidate.key] = p
                elif candidate.description == "The proposition is false.":
                    expected[candidate.key] = 1 - p
                else:
                    raise ValueError("unknown truth rubric")
            error = max(abs(expected[key] - record.targets[name][key]) for key in expected)
            maximum_error = max(maximum_error, error)
            if error > 1e-12:
                raise ValueError("rendered question disagrees with target: " + record.id)
            target, valid, _ = canonical_target(record)
            ordered = (
                sorted(candidates, key=lambda c: c.description)
                if question.type == "choice"
                else candidates
            )
            if int(valid.sum()) != 2:
                raise ValueError("incorrect canonical target width")
            if max(abs(float(target[i]) - expected[c.key]) for i, c in enumerate(ordered)) > 1e-7:
                raise ValueError("canonical training target projection changed")
            counts[f"{split}/{kind}/{question.type}"] += 1
    if sum(counts.values()) != 16080:
        raise ValueError("incomplete skill target audit")
    result = {
        "passed": True,
        "questions_checked": sum(counts.values()),
        "maximum_rendered_target_error": maximum_error,
        "question_counts": dict(counts),
        "original_groups_splits_families_preserved": True,
        "canonical_targets_agree": True,
        "calibration_or_final_used": False,
        "model_inference_used": False,
        "source_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("scripts/verify_workspace_calibration_data.py"),
                Path("src/veyra/workflow_facts.py"),
                Path("scripts/workflow_skill_common.py"),
            )
        },
        "data_audit_sha256": digest(root / "audit.json"),
        "scope": (
            "Propositions and rubric direction re-parsed independently of builder annotations; "
            "primitive and Boolean parsers are shared with prior audited sources."
        ),
    }
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
