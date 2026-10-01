"""Independently recompute fresh final targets from rendered inputs, without inference."""

import argparse
import ast
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

from train_foundation_head import write

from veyra.data import read_records
from veyra.workflow_facts import stated_probabilities


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expression(text, facts):
    count = re.fullmatch(r"at least (\d+) of \(([^()]+)\)", text)
    if count:
        return sum(facts[token.strip()] for token in count[2].split(",")) >= int(count[1])
    tree = ast.parse(
        text.replace("AND", "and").replace("OR", "or").replace("NOT", "not"), mode="eval"
    )

    def visit(node):
        if isinstance(node, ast.Name) and node.id in facts:
            return facts[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not visit(node.operand)
        if isinstance(node, ast.BoolOp):
            values = [visit(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
        raise ValueError("unsupported rendered Boolean expression")

    return visit(tree.body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["output"])
    output = root / "independent-target-audit.json"
    if output.exists():
        raise FileExistsError("data audit is immutable")
    records = read_records(root / "records.jsonl")
    old = read_records(Path(config["old_records"]))
    exclusions = [*old, *(r for p in config["exclude_records"] for r in read_records(Path(p)))]
    selected = [record for record in records if record.split == "test"]
    old_ids = {record.group_id for record in exclusions}
    if any(record.group_id in old_ids for record in selected):
        raise ValueError("fresh calibration includes a previous group")
    maximum_error, checked, domains = 0.0, 0, Counter()
    for record in selected:
        domain = next(tag[7:] for tag in record.tags if tag.startswith("domain:"))
        domains[domain] += 1
        if domain not in {"workflow_new", "workflow_known", "uncertainty"}:
            continue
        text = record.request.state.text
        probabilities = stated_probabilities(text)
        name, question = next(iter(record.request.questions.items()))
        if domain == "workflow_known":
            facts = dict(zip("ABCD", map(bool, probabilities)))
            if any(value not in (0, 1) for value in probabilities):
                raise ValueError("fresh explicit policy unexpectedly hides a fact")
            if question.type == "noul":
                policy = question.instructions.split("exactly when ", 1)[1].split(" is true.", 1)[0]
                value = expression(policy, facts)
                target = {"false": float(not value), "true": float(value)}
            else:
                criteria = (
                    question.criteria
                    if question.type == "choice"
                    else dict(enumerate(question.criteria))
                )
                target = {
                    str(key): float(expression(meaning.split(": ", 1)[1].removesuffix("."), facts))
                    for key, meaning in criteria.items()
                }
                if sum(target.values()) != 1:
                    raise ValueError("rendered explicit policy is not mutually exclusive")
        else:
            rules = re.findall(r"^\d+\. IF (.+) THEN (PROCEED|REVIEW|HOLD)\.$", text, re.MULTILINE)
            default = re.search(r"^Otherwise (PROCEED|REVIEW|HOLD)\.$", text, re.MULTILINE)
            if not rules or not default:
                raise ValueError("no auditable priority policy")
            distribution = dict.fromkeys(("PROCEED", "REVIEW", "HOLD"), 0.0)
            for assignment in itertools.product((False, True), repeat=4):
                facts = dict(zip("ABCD", assignment))
                mass = 1.0
                for value, probability in zip(assignment, probabilities):
                    mass *= probability if value else 1 - probability
                outcome = next(
                    (action for policy, action in rules if expression(policy, facts)), default[1]
                )
                distribution[outcome] += mass
            if question.type == "noul":
                target = {"false": 1 - distribution["HOLD"], "true": distribution["HOLD"]}
            else:
                criteria = (
                    question.criteria
                    if question.type == "choice"
                    else dict(enumerate(question.criteria))
                )
                target = {
                    str(key): distribution[meaning.removeprefix("Outcome ").removesuffix(".")]
                    for key, meaning in criteria.items()
                }
        if set(target) != set(record.targets[name]):
            raise ValueError("recomputed target keys differ")
        error = max(abs(value - record.targets[name][key]) for key, value in target.items())
        maximum_error = max(maximum_error, error)
        if error > 1e-7:
            raise ValueError("rendered rule does not match target: " + record.id)
        checked += 1
    if checked != 1920:
        raise ValueError("incomplete fresh policy target audit")
    preserved = {}
    for split in ("train", "dev", "calibration"):
        before = "".join(record.model_dump_json() + "\n" for record in old if record.split == split)
        after = "".join(
            record.model_dump_json() + "\n" for record in records if record.split == split
        )
        if before != after:
            raise ValueError("a frozen data split changed")
        preserved[split] = hashlib.sha256(before.encode()).hexdigest()
    result = {
        "passed": True,
        "records_sha256": digest(root / "records.jsonl"),
        "final_design_sha256": digest(args.config),
        "excluded_record_files": {p: digest(p) for p in config["exclude_records"]},
        "source_sha256": digest(Path(__file__)),
        "primitive_parser_sha256": digest("src/veyra/workflow_facts.py"),
        "fresh_policy_targets_recomputed": checked,
        "maximum_target_absolute_error": maximum_error,
        "fresh_final_domain_counts": dict(domains),
        "preserved_subset_sha256": preserved,
        "previous_group_overlap": 0,
        "model_inference_used": False,
        "scope": (
            "Rendered rule targets independently parsed; public hard labels retain their "
            "source provenance. This does not validate empirical business frequencies."
        ),
    }
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
