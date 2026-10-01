"""Construct fresh final-only families and public groups; preserve train/dev/calibration."""

import argparse
import hashlib
import itertools
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from build_foundation_data import image_signature
from build_workflow_v12 import ACTION, decision, distribution, wording
from train_foundation_head import write

from veyra.data import Source, TrainingRecord, audit_records, read_records
from veyra.schema import DecisionRequest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_state(text):
    return " ".join(re.sub(r"Case reference [0-9a-f]+", "", text).lower().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config
    config = read(config_path)

    def stable(text):
        return hashlib.sha256((config["identity_namespace"] + ":" + text).encode()).hexdigest()

    old_path, root = Path(config["old_records"]), Path(config["output"])
    if root.exists():
        raise FileExistsError("fresh calibration corpus is immutable")
    if digest(old_path) != config["old_records_sha256"]:
        raise ValueError("original split data changed")
    source_path = Path(config["source_output"]) / "retention-selections.json"
    selected = read(source_path)
    if (
        selected["config_sha256"] != digest(config_path)
        or selected["no_model_outputs_used"] is not True
    ):
        raise ValueError("fresh public selection did not use the declared protocol")
    for path, expected in selected["source_files"].items():
        if digest(path) != expected:
            raise ValueError("public source changed: " + path)
    original_downloads = read("data/foundation-v11-source/downloads.json")
    for name in ("snli-validation.parquet", "banking-train.csv"):
        if digest(Path("data/foundation-v11-source") / name) != original_downloads[name]["sha256"]:
            raise ValueError("public text source differs from its pinned download")
    cifar_download = read("data/workflow-v12-source/downloads.json")
    if (
        digest("data/workflow-v12-source/cifar-test.parquet")
        != cifar_download["files"]["cifar-test.parquet"]["sha256"]
    ):
        raise ValueError("CIFAR source differs from its pinned download")
    old = read_records(old_path)
    foundation = read_records(Path("data/foundation-v11/records.jsonl"))
    excluded = [(read_records(Path(p)), Path(p)) for p in config["exclude_records"]]
    previous_populations = [
        (old, old_path),
        (foundation, Path("data/foundation-v11/records.jsonl")),
        *excluded,
    ]
    previous_records = [r for population, _ in previous_populations for r in population]
    records = [record for record in old if record.split != "test"]
    original_groups = {record.group_id for record in previous_records}
    original_states = {
        normalized_state(r.request.state.text)
        for r in previous_records
        if not r.request.state.images
    }
    root.mkdir()
    (root / "images").mkdir()
    for record in records:
        for image in record.request.state.images:
            origin, target = old_path.parent / image.path, root / image.path
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(origin, target)
    source = Source(
        id=config["protocol"],
        revision="sha256:" + digest(Path(__file__)),
        license="Apache-2.0",
        url="https://github.com/ken-jo/veyra",
    )
    fresh, latent = [], []

    def add(
        group, family, name, state, question, probabilities, tags, origin=source, image_sha=None
    ):
        if group in original_groups:
            raise ValueError("a calibration group was reused")
        if not state.get("images") and normalized_state(state["text"]) in original_states:
            raise ValueError("a calibration state was reused, ignoring case nonce")
        gold = (
            ["gold_key:" + max(probabilities, key=probabilities.get)]
            if max(probabilities.values()) >= 1 - 1e-9
            else []
        )
        fresh.append(
            TrainingRecord(
                id=group + ":" + name,
                group_id=group,
                split="test",
                family=family,
                language="en",
                source=origin,
                request=DecisionRequest.model_validate(
                    {"state": state, "questions": {name: question}}
                ),
                targets={name: probabilities},
                tags=[*tags, *gold],
                image_sha256=image_sha,
            )
        )

    def typed(group, family, name, text, kind, outcome, tags):
        if kind == "noul":
            instructions = (
                "Under the supplied rules and stated uncertainty, "
                "what is the probability that HOLD is required?"
                if name == "uncertain"
                else "Under the supplied rules, is HOLD the required outcome?"
            )
            question = {"type": kind, "instructions": instructions}
            target, critical = {"false": 1 - outcome[2], "true": outcome[2]}, "true"
        else:
            meanings = [f"Outcome {action}." for action in ACTION]
            if kind == "score":
                question = {
                    "type": kind,
                    "instructions": "Apply the supplied rules and return the distribution "
                    "over the ordered outcomes.",
                    "criteria": meanings,
                }
                target, critical = dict(zip(map(str, range(3)), outcome)), "2"
            else:
                keys = ["candidate_" + stable(group + str(i))[:10] for i in range(3)]
                question = {
                    "type": kind,
                    "instructions": "Which outcome follows from the evidence and any "
                    "supplied rules? "
                    "Return probabilities when facts are uncertain.",
                    "criteria": dict(zip(keys, meanings)),
                }
                target, critical = dict(zip(keys, outcome)), keys[2]
        add(
            group,
            family,
            name,
            {"text": text},
            question,
            target,
            [*tags, "critical_key:" + critical],
        )

    # New final-only family definitions were declared before any recovery policy or final output.
    for family_index, (family, role, fields, rules, default) in enumerate(config["final_families"]):
        if role != "test":
            raise ValueError("Fresh final families must have the test-only role")
        for index in range(config["quotas"]["procedural_workflow_groups_per_train_or_dev_family"]):
            group = f"{config['group_namespace']}:{family}:test:{index:04}"
            rng = random.Random(int(stable(group)[:16], 16))
            possible = [
                facts
                for facts in itertools.product((False, True), repeat=4)
                if decision(rules, default, facts) == index % 3
            ]
            facts = rng.choice(possible)
            thresholds = [rng.randint(10, 90) for _ in fields]
            readings = [
                rng.randint(t, 100) if flag else rng.randint(0, t - 1)
                for t, flag in zip(thresholds, facts)
            ]
            definitions = [
                f"{letter} means {field} >= {threshold}."
                for letter, field, threshold in zip("ABCD", fields, thresholds)
            ]
            rng.shuffle(definitions)
            policy = (
                "Apply the first matching rule in the listed priority order.\n"
                + "\n".join(
                    f"{i + 1}. IF {wording(condition)} THEN {ACTION[outcome]}."
                    for i, (condition, outcome) in enumerate(rules)
                )
                + f"\nOtherwise {ACTION[default]}."
            )
            prefix = (
                f"Workflow: {family.replace('_', ' ')}. Case reference {stable(group)[:10]}.\n"
                + "\n".join(definitions)
                + "\n"
                + policy
            )
            text = prefix + "\nObserved measurements: " + json.dumps(dict(zip(fields, readings)))
            outcome = [float(i == index % 3) for i in range(3)]
            for kind in ("choice", "score", "noul"):
                typed(
                    group,
                    family,
                    kind,
                    text,
                    kind,
                    outcome,
                    [
                        "domain:workflow_new",
                        "view:complete",
                        "family_role:" + role,
                        "rule_depth:" + str(len(rules)),
                        "fresh_final",
                    ],
                )
            condition = ("missing", "conflicting", "shifted_prior")[(index + family_index) % 3]
            kind = ("choice", "score", "noul")[(index // 3 + family_index) % 3]
            hidden = set(rng.sample(range(4), 2))
            priors = [
                rng.choice(
                    [0.05, 0.1, 0.9, 0.95]
                    if condition == "shifted_prior"
                    else [0.2, 0.35, 0.65, 0.8]
                )
                for _ in fields
            ]
            conflict = min(hidden)
            for _ in range(2):
                posterior = [priors[i] if i in hidden else float(facts[i]) for i in range(4)]
                if condition == "conflicting":
                    prior = priors[conflict]
                    posterior[conflict] = (
                        prior * 0.8 * 0.35 / (prior * 0.8 * 0.35 + (1 - prior) * 0.2 * 0.65)
                    )
                target = distribution(rules, default, posterior)
                if (kind == "noul" and 0.001 < target[2] < 0.999) or (
                    kind != "noul" and max(target) < 0.999
                ):
                    break
                hidden = set(range(4))
            observed = {
                field: ("unobserved" if i in hidden else readings[i])
                for i, field in enumerate(fields)
            }
            text = prefix + "\nObserved measurements: " + json.dumps(observed)
            text += (
                "\nFor this question, unknown conditions are independent Bernoulli variables "
                "with these specified prior probabilities: "
                + json.dumps({"ABCD"[i]: priors[i] for i in sorted(hidden)})
                + ". All unreported information remains unknown."
            )
            if condition == "conflicting":
                text += (
                    f"\nTwo independent sensors report on condition {'ABCD'[conflict]}. "
                    "Sensor X says TRUE and is correct with probability 0.80. "
                    "Sensor Y says FALSE and is correct with probability 0.65. "
                    "Each sensor has equal sensitivity and specificity. "
                    "Their errors are independent conditional on the true condition. "
                    "Use the reports and prior together."
                )
            typed(
                group,
                family,
                "uncertain",
                text,
                kind,
                target,
                [
                    "domain:uncertainty",
                    "view:partial_evidence",
                    "known_conditional_distribution",
                    "condition:" + condition,
                    "family_role:" + role,
                    "fresh_final",
                ],
            )
            latent.append(
                {
                    "group": group,
                    "condition": condition,
                    "posterior": posterior,
                    "outcome_distribution": target,
                }
            )

    # Newly authored explicit policies, not a relabeling or rewrite of the public states.
    templates = defaultdict(dict)
    for record in old:
        if record.split == "calibration" and "domain:workflow_known" in record.tags:
            for name, question in record.request.questions.items():
                templates[record.family][name] = question.type
    fields_by_family = {
        "workflow_agent_trace_observability": [
            "evidence_completeness",
            "tool_failure",
            "scope_breach",
            "irreversible_action",
        ],
        "workflow_customer_service": [
            "account_impact",
            "wait_time",
            "unresolved_requests",
            "service_interruption",
        ],
        "workflow_invoice_processing": [
            "amount_variance",
            "delivery_variance",
            "duplicate_signal",
            "approval_deficit",
        ],
        "workflow_security_incidents": [
            "credential_signal",
            "host_anomaly",
            "external_exposure",
            "privilege_change",
        ],
    }
    known_labels = defaultdict(Counter)
    for family, question_types in sorted(templates.items()):
        fields = fields_by_family[family]
        assignments = list(itertools.product((False, True), repeat=4))
        random.Random(int(stable(family)[:16], 16)).shuffle(assignments)
        for index in range(config["quotas"]["known_workflow_groups_per_family"]):
            group = f"{config['group_namespace']}:known:{family}:{index:04}"
            rng = random.Random(int(stable(group)[:16], 16))
            facts = assignments[index % len(assignments)]
            thresholds = [rng.randint(10, 90) for _ in fields]
            readings = [
                rng.randint(t, 100) if flag else rng.randint(0, t - 1)
                for t, flag in zip(thresholds, facts)
            ]
            definitions = [
                f"{letter} means {field} >= {threshold}."
                for letter, field, threshold in zip("ABCD", fields, thresholds)
            ]
            rng.shuffle(definitions)
            text = (
                f"Workflow: {family.removeprefix('workflow_').replace('_', ' ')}. "
                f"Case reference {stable(group)[:10]}.\n"
                + "\n".join(definitions)
                + "\nObserved measurements: "
                + json.dumps(dict(zip(fields, readings)))
            )
            for question_index, (name, kind) in enumerate(sorted(question_types.items())):
                left, right = question_index % 4, (question_index + 1) % 4
                a, b, c = "ABCD"[left], "ABCD"[right], "ABCD"[(question_index + 2) % 4]
                if kind == "noul":
                    expression = f"({a} AND {b}) OR (NOT {c})"
                    value = (facts[left] and facts[right]) or not facts[(question_index + 2) % 4]
                    question = {
                        "type": kind,
                        "instructions": (
                            f"Apply this caller-supplied policy for {name.replace('_', ' ')}. "
                            f"The statement is true exactly when {expression} is true. "
                            "Conditions are defined by the measured values in the evidence."
                        ),
                        "criteria": {
                            "false": "The stated Boolean condition is false.",
                            "true": "The stated Boolean condition is true.",
                        },
                    }
                    target = {"false": float(not value), "true": float(value)}
                else:
                    value = 2 * int(facts[left]) + int(facts[right])
                    expressions = [
                        f"NOT {a} AND NOT {b}",
                        f"NOT {a} AND {b}",
                        f"{a} AND NOT {b}",
                        f"{a} AND {b}",
                    ]
                    meanings = [
                        f"Policy {'level' if kind == 'score' else 'outcome'} {i}: {expression}."
                        for i, expression in enumerate(expressions)
                    ]
                    keys = (
                        list(map(str, range(4)))
                        if kind == "score"
                        else ["option_" + stable(group + name + str(i))[:10] for i in range(4)]
                    )
                    question = {
                        "type": kind,
                        "instructions": (
                            f"Apply this caller-supplied policy for {name.replace('_', ' ')}. "
                            "Exactly one of the four Boolean conditions in the criteria holds. "
                            "Use the condition definitions and measurements in the evidence."
                        ),
                        "criteria": meanings if kind == "score" else dict(zip(keys, meanings)),
                    }
                    target = {key: float(i == value) for i, key in enumerate(keys)}
                known_labels[family + "/" + name][str(value)] += 1
                add(
                    group,
                    family,
                    name,
                    {"text": text},
                    question,
                    target,
                    [
                        "domain:workflow_known",
                        "view:explicit_policy",
                        "fresh_final",
                        "exact_policy_target",
                        "family_role:known",
                        "new_policy_not_original_teacher_label",
                    ],
                )

    for row in selected["text_rows"]:
        group = config["group_namespace"] + ":" + row["source"] + ":" + row["id"]
        if row["source"] == "snli":
            meanings = [
                "The hypothesis follows from the premise.",
                "The premise does not determine whether the hypothesis is true.",
                "The hypothesis contradicts the premise.",
            ]
            label, domain = row["label"], "text_nli"
            origin = Source(
                id="stanfordnlp/snli",
                revision="cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
                license="CC-BY-SA-4.0",
                url="https://huggingface.co/datasets/stanfordnlp/snli",
                upstream_split="validation",
            )
        else:
            rng = random.Random(int(stable(group)[:16], 16))
            options = [row["label"]] + rng.sample(
                [x for x in selected["banking_categories"] if x != row["label"]], 7
            )
            rng.shuffle(options)
            meanings = ["The user's request concerns " + x.replace("_", " ") + "." for x in options]
            label, domain = options.index(row["label"]), "text_intent"
            origin = Source(
                id="PolyAI/banking77",
                revision="57ec275d8078af65b7731c2a98be812d844a6d6b",
                license="CC-BY-4.0",
                url="https://huggingface.co/datasets/PolyAI/banking77",
                upstream_split="train",
            )
        keys = ["candidate_" + stable(group + str(i))[:10] for i in range(len(meanings))]
        question = {
            "type": "choice",
            "instructions": "Which outcome follows from the evidence and any supplied rules? "
            "Return probabilities when facts are uncertain.",
            "criteria": dict(zip(keys, meanings)),
        }
        add(
            group,
            domain,
            "decision",
            {"text": row["text"]},
            question,
            {key: float(i == label) for i, key in enumerate(keys)},
            ["domain:" + domain, "view:recognition", "fresh_retention", "fresh_final"],
            origin,
        )

    signatures, existing_images = [], set()
    for population, path in previous_populations:
        for record in population:
            if record.image_sha256 and record.image_sha256 not in existing_images:
                existing_images.add(record.image_sha256)
                signatures.append(
                    image_signature(
                        (path.parent / record.request.state.images[0].path).read_bytes()
                    )
                )
    initial_signature_count = len(signatures)
    download = read("data/workflow-v12-source/downloads.json")
    origin = Source(
        id="uoft-cs/cifar10",
        revision=download["revision"],
        license="unknown",
        url="https://huggingface.co/datasets/uoft-cs/cifar10",
        upstream_split="test",
    )
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    counts, excluded = Counter(), 0
    for row in sorted(selected["image_candidates"], key=lambda r: stable(r["sha256"])):
        label = row["label"]
        if counts[label] >= config["quotas"]["cifar10_per_class"]:
            continue
        source_image = Path(row["path"])
        if digest(source_image) != row["sha256"] or row["sha256"] in existing_images:
            raise ValueError("new image identity mismatch or overlap")
        signature = image_signature(source_image.read_bytes())
        pixel, phash, dhash, color = signature
        if any(
            pixel == p
            or (
                (phash ^ ph).bit_count() <= 4
                and (dhash ^ dh).bit_count() <= 8
                and max(abs(a - b) for a, b in zip(color, c)) <= 35
            )
            for p, ph, dh, c in signatures
        ):
            excluded += 1
            continue
        signatures.append(signature)
        counts[label] += 1
        destination = root / "images" / (row["sha256"] + ".png")
        os.link(source_image, destination)
        group = config["group_namespace"] + ":cifar:" + row["sha256"][:24]
        keys = ["candidate_" + stable(group + str(i))[:10] for i in range(10)]
        question = {
            "type": "choice",
            "instructions": "Which outcome follows from the evidence and any supplied rules? "
            "Return probabilities when facts are uncertain.",
            "criteria": {
                key: "The main object is a " + labels[i] + "." for i, key in enumerate(keys)
            },
        }
        add(
            group,
            "photo_guard",
            "decision",
            {
                "text": "Identify the main object in the photograph.",
                "images": [{"path": "images/" + destination.name}],
            },
            question,
            {key: float(i == label) for i, key in enumerate(keys)},
            [
                "domain:photo_guard",
                "view:recognition",
                "fresh_retention",
                "evaluation_only",
                "fresh_final",
            ],
            origin,
            row["sha256"],
        )
    if any(counts[i] != config["quotas"]["cifar10_per_class"] for i in range(10)):
        raise ValueError("insufficient independent photographs")
    counts_by_type = Counter(next(iter(record.request.questions.values())).type for record in fresh)
    if (
        len(fresh) != config["expected_calibration_questions"]
        or len({r.group_id for r in fresh}) != config["expected_calibration_groups"]
        or dict(counts_by_type) != config["expected_type_counts"]
    ):
        raise ValueError("fresh calibration population changed")
    if any(len(counts) < 2 for counts in known_labels.values()):
        raise ValueError("known-workflow question collapsed onto one label")
    records.extend(fresh)
    path = root / "records.jsonl"
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    summary = audit_records(read_records(path))
    subset_hashes = {}
    for split in ("train", "dev", "calibration"):
        before = "".join(r.model_dump_json() + "\n" for r in old if r.split == split).encode()
        after = "".join(r.model_dump_json() + "\n" for r in records if r.split == split).encode()
        if before != after:
            raise ValueError("a frozen training/development/final record changed")
        subset_hashes[split] = hashlib.sha256(before).hexdigest()
    write(root / "conditional-target-audit.json", latent)
    protocol = {
        **summary,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_sha256": digest(path),
        "final_design_sha256": digest(config_path),
        "identity_namespace": config["identity_namespace"],
        "excluded_record_files": {str(p): digest(p) for _, p in previous_populations},
        "study_protocol_sha256": digest(config["study_protocol"]),
        "release_protocol_sha256": digest(config["release_protocol"]),
        "generator_sha256": digest(Path(__file__)),
        "procedural_rule_source_sha256": digest(Path("scripts/build_workflow_v12.py")),
        "retention_selection_sha256": digest(source_path),
        "preserved_subset_sha256": subset_hashes,
        "final_type_counts": dict(counts_by_type),
        "fresh_final_groups": len({r.group_id for r in fresh}),
        "final_group_overlap_with_previous": 0,
        "final_state_overlap_ignoring_case_nonce": 0,
        "images_screened_against_previous_unique_images": initial_signature_count,
        "near_duplicate_image_candidates_excluded": excluded,
        "new_known_workflow_label_counts": dict(known_labels),
        "known_workflow_labels_are_new_explicit_policy_targets": True,
        "old_calibration_policy_regression_required": True,
        "model_outputs_used": False,
        "final_inference_performed": False,
    }
    write(root / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "records": summary["records"],
                "fresh_final": len(fresh),
                "preserved_splits": list(subset_hashes),
                "records_sha256": protocol["records_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
