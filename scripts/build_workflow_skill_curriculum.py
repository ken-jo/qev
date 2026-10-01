"""Create direct prerequisite questions from preserved training/development states only."""

import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from train_foundation_head import write

from veyra.data import TrainingRecord, audit_records
from veyra.workflow_facts import stated_probabilities

SOURCE = Path("data/workflow-v12/records.jsonl")
SOURCE_SHA = "145e01a41ed73f9c02524e5bf9a8a0efa8b73300eb9871b30ab0173b29ab5571"
EXECUTION = Path("data/workflow-v12-execution-targets/targets.jsonl")
EXECUTION_SHA = "7cfbbb767c27e37b4de360036fdbf0525dc56ac4eda4aa0a056e9e2137f35867"
OUTPUT = Path("data/workflow-v12-skills")
SEED = 193


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_record(base, state_hash, kind, index, probability, rng):
    raw_probability = probability
    if not -1e-12 <= probability <= 1 + 1e-12:
        raise ValueError("execution marginal is outside probability range")
    # Enumerating sixteen worlds can exceed one by floating-point roundoff only.
    probability = max(0.0, min(1.0, probability))
    negate = bool(rng.randrange(2))
    assertion = "false" if negate else "true"
    if kind == "fact":
        proposition = f"Condition {'ABCD'[index]}, as defined in the evidence, is {assertion}."
        clarification = "Judge this condition alone, without applying the outcome rules."
    else:
        proposition = f"The Boolean condition of rule {index + 1} is {assertion}."
        clarification = (
            "Evaluate that rule's condition alone, regardless of whether an earlier rule matches; "
            "do not judge whether this is the first matching rule."
        )
    p = 1 - probability if negate else probability
    instructions = (
        f"Proposition: {proposition} {clarification} Use the observed measurements and any stated "
        "priors or sensor evidence. Return the distribution over the supplied criteria."
    )
    question_type = ("choice", "score", "noul")[rng.randrange(3)]
    true_text, false_text = "The proposition is true.", "The proposition is false."
    question = {"type": question_type, "instructions": instructions}
    if question_type == "choice":
        keys = ["candidate_" + f"{rng.getrandbits(32):08x}" for _ in range(2)]
        if keys[0] == keys[1]:
            raise ValueError("candidate identity collision")
        pairs = [(keys[0], true_text, p), (keys[1], false_text, 1 - p)]
        rng.shuffle(pairs)
        question["criteria"] = {key: description for key, description, _ in pairs}
        target = {key: mass for key, _, mass in pairs}
    elif question_type == "score":
        pairs = [(true_text, p), (false_text, 1 - p)]
        rng.shuffle(pairs)
        question["criteria"] = [description for description, _ in pairs]
        target = {str(i): mass for i, (_, mass) in enumerate(pairs)}
    else:
        question["criteria"] = {"true": true_text, "false": false_text}
        target = {"true": p, "false": 1 - p}
    identifier = f"skill12:{state_hash[:18]}:{kind}:{index}"
    tags = [
        f"domain:skill_{kind}",
        "skill_only",
        "polarity:negated" if negate else "polarity:positive",
    ]
    tags += [tag for tag in base.tags if tag.startswith(("view:", "condition:"))]
    if p in (0, 1):
        tags.append("gold_key:" + max(target, key=target.get))
    record = TrainingRecord.model_validate(
        {
            "id": identifier,
            "group_id": base.group_id,
            "split": base.split,
            "family": base.family,
            "language": "en",
            "request": {
                "state": base.request.state.model_dump(),
                "questions": {"decision": question},
            },
            "targets": {"decision": target},
            "source": {
                "id": "veyra-workflow-prerequisite-curriculum",
                "revision": "sha256:" + digest(__file__),
                "license": "Apache-2.0",
                "url": "https://github.com/ken-jo/veyra",
            },
            "tags": tags,
        }
    )
    annotation = {
        "id": identifier,
        "source_record": base.id,
        "state_sha256": state_hash,
        "group_id": base.group_id,
        "split": base.split,
        "kind": kind,
        "index": index,
        "negated": negate,
        "proposition_true_probability": p,
        "original_condition_probability": probability,
        "roundoff_clamp_absolute_change": abs(probability - raw_probability),
    }
    return record, annotation


def main():
    if OUTPUT.exists():
        raise FileExistsError("skill curriculum data are immutable")
    if digest(SOURCE) != SOURCE_SHA or digest(EXECUTION) != EXECUTION_SHA:
        raise ValueError("source or independently audited execution annotations changed")
    execution = {}
    for line in EXECUTION.open(encoding="utf-8"):
        item = json.loads(line)
        if item["split"] not in {"train", "dev"}:
            raise ValueError("execution annotations include held-out data")
        execution[item["state_sha256"]] = item
    states = {}
    for line in SOURCE.open(encoding="utf-8"):
        raw = json.loads(line)
        if raw["split"] not in {"train", "dev"}:
            continue
        if not {"domain:workflow_new", "domain:uncertainty"}.intersection(raw["tags"]):
            continue
        base = TrainingRecord.model_validate(raw)
        state_hash = hashlib.sha256(base.request.state.text.encode()).hexdigest()
        states.setdefault(state_hash, base)
    if Counter(r.split for r in states.values()) != {"train": 1920, "dev": 600}:
        raise ValueError("unexpected original state population")
    records, annotations, diagnostic_ids = [], [], []
    counters = Counter()
    for state_hash, base in states.items():
        annotation = execution[state_hash]
        if annotation["group_id"] != base.group_id or annotation["split"] != base.split:
            raise ValueError("inconsistent annotation identity")
        facts = stated_probabilities(base.request.state.text)
        if (
            max(abs(a - b) for a, b in zip(facts, annotation["fact_probabilities"], strict=True))
            > 1e-9
        ):
            raise ValueError("independent fact annotations disagree")
        rules = re.findall(
            r"^\d+\. IF (.+) THEN (?:PROCEED|REVIEW|HOLD)\.$", base.request.state.text, re.M
        )
        if len(rules) != len(annotation["rule_satisfaction_probabilities"]):
            raise ValueError("rule identity mismatch")
        rng = random.Random(f"{SEED}:{state_hash}")
        state_records = {"fact": [], "rule": []}
        for kind, values in (
            ("fact", facts),
            ("rule", annotation["rule_satisfaction_probabilities"]),
        ):
            for index, probability in enumerate(values):
                record, detail = make_record(base, state_hash, kind, index, probability, rng)
                records.append(record)
                annotations.append(detail)
                state_records[kind].append(record)
        if base.split == "dev":
            # One fact and one rule per state, fixed before any model predictions.
            for kind in ("fact", "rule"):
                options = state_records[kind]
                diagnostic_ids.append(options[counters[kind] % len(options)].id)
                counters[kind] += 1
    summary = audit_records(records)
    if len(diagnostic_ids) != 1200 or len(set(diagnostic_ids)) != 1200:
        raise ValueError("unexpected diagnostic sample")
    group_splits = {r.group_id: r.split for r in records}
    if len(group_splits) != 1260:
        raise ValueError("skill queries must preserve the original 960/300 groups")
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source_sha256": {
            str(p): digest(p)
            for p in (SOURCE, EXECUTION, Path(__file__), Path("src/veyra/workflow_facts.py"))
        },
        "scope": (
            "Prospective train/development prerequisite questions; no model training "
            "or calibration/final inference."
        ),
        "data_policy": (
            "Original states are preserved exactly. Every derived view inherits its source "
            "group, split and family. Direct fact truth and rule-condition marginals come from "
            "the independently audited 16-world execution annotations. No final or calibration "
            "state is interpreted."
        ),
        "question_design": (
            "One question per fact and per rule condition, randomized proposition polarity, "
            "caller-defined choice keys, randomized score rubric direction, or noul. Rule truth "
            "is independent of first-match priority. No intermediate target enters the input."
        ),
        "development_diagnostic": (
            "One fact and one rule question per each of 600 development states; fixed before "
            "model predictions; not a release criterion."
        ),
        "calibration_or_final_used": False,
        "runtime_solver_added": False,
        "training_protocol_required": True,
    }
    write(OUTPUT / "protocol.json", protocol)
    for split in ("train", "dev"):
        (OUTPUT / f"{split}.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in records if r.split == split),
            encoding="utf-8",
            newline="\n",
        )
    (OUTPUT / "annotations.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in annotations), encoding="utf-8", newline="\n"
    )
    write(OUTPUT / "diagnostic-ids.json", diagnostic_ids)
    audit = {
        "passed": True,
        "summary": summary,
        "questions_by_split_and_skill": dict(
            Counter(f"{r.split}/{a['kind']}" for r, a in zip(records, annotations, strict=True))
        ),
        "groups_by_split": dict(Counter(group_splits.values())),
        "state_text_unchanged": all(
            hashlib.sha256(r.request.state.text.encode()).hexdigest() == a["state_sha256"]
            for r, a in zip(records, annotations, strict=True)
        ),
        "diagnostic_questions": len(diagnostic_ids),
        "maximum_roundoff_clamp": max(a["roundoff_clamp_absolute_change"] for a in annotations),
        "files_sha256": {p.name: digest(p) for p in sorted(OUTPUT.iterdir())},
        "calibration_or_final_used": False,
        "release_allowed": False,
    }
    if not audit["state_text_unchanged"]:
        raise ValueError("skill input states changed")
    write(OUTPUT / "audit.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
