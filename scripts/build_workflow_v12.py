"""Build frozen workflow, exact uncertainty and independent retention groups."""

import hashlib
import itertools
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_foundation_data import image_signature

from veyra.data import Source, TrainingRecord, audit_records, read_records
from veyra.schema import DecisionRequest

ROOT = Path("data/workflow-v12")
ACTION = ["PROCEED", "REVIEW", "HOLD"]
# Family-level rule compositions are held out, not only candidate identifiers.
FAMILIES = [
    (
        "shipping_release",
        "train",
        ["payment", "stock", "address", "restriction"],
        [(3, 2), (("and", 0, 1, 2), 0), (("and", 0, 1), 1)],
        2,
    ),
    (
        "staff_access",
        "train",
        ["training", "approval", "contract", "suspension"],
        [(3, 2), (("and", 0, 1, 2), 0), (2, 1)],
        2,
    ),
    (
        "purchase_return",
        "train",
        ["receipt", "return_window", "damage", "warranty"],
        [(("and", 0, 1), 0), (("and", 2, 3), 1)],
        2,
    ),
    (
        "backup_restore",
        "train",
        ["checksum", "replica", "approval", "incident"],
        [(("and", ("not", 0), ("not", 1)), 2), (("and", 2, ("or", 0, 1)), 0), (3, 2)],
        1,
    ),
    (
        "content_publish",
        "train",
        ["review", "rights", "embargo", "correction"],
        [(2, 2), (("and", 0, 1, ("not", 3)), 0)],
        1,
    ),
    (
        "hardware_dispatch",
        "train",
        ["test", "inventory", "permit", "alarm"],
        [(("and", 3, ("not", 2)), 2), (("and", 0, 1), 0)],
        1,
    ),
    (
        "grant_payment",
        "dev",
        ["merit", "budget", "approval", "conflict"],
        [(3, 2), (("count", 2, 0, 1, 2), 0)],
        1,
    ),
    (
        "warehouse_transfer",
        "dev",
        ["capacity", "inspection", "reservation", "quarantine"],
        [(("and", 3, ("or", ("not", 0), ("not", 1))), 2), (("and", ("or", 0, 2), 1), 0)],
        1,
    ),
    (
        "service_repair",
        "dev",
        ["diagnosis", "parts", "customer_consent", "hazard"],
        [(("and", 3, ("not", 0)), 2), (("and", 2, ("or", 0, 1)), 0)],
        1,
    ),
    (
        "laboratory_batch",
        "test",
        ["purity", "traceability", "supervision", "contamination"],
        [(("and", 3, ("not", 2)), 2), (("and", ("or", 0, 2), ("or", 1, 2), ("not", 3)), 0)],
        1,
    ),
    (
        "archive_loan",
        "test",
        ["condition", "insurance", "authorization", "restriction"],
        [(("and", 3, ("or", ("not", 1), ("not", 2))), 2), (("and", 2, ("or", 0, 1)), 0)],
        1,
    ),
    (
        "utility_maintenance",
        "test",
        ["isolation", "crew", "permit", "emergency"],
        [
            (("and", ("not", 0), ("or", 3, ("not", 2))), 2),
            (("and", 0, ("or", 1, ("and", 2, 3))), 0),
        ],
        1,
    ),
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stable(text):
    return digest(("veyra-workflow-v12-131:" + text).encode())


def truth(expression, facts):
    if isinstance(expression, int):
        return facts[expression]
    op, *terms = expression
    if op == "not":
        return not truth(terms[0], facts)
    if op == "count":
        return sum(truth(t, facts) for t in terms[1:]) >= terms[0]
    values = [truth(t, facts) for t in terms]
    return all(values) if op == "and" else any(values)


def wording(expression):
    if isinstance(expression, int):
        return "ABCD"[expression]
    op, *terms = expression
    if op == "not":
        return "NOT " + wording(terms[0])
    if op == "count":
        return f"at least {terms[0]} of (" + ", ".join(wording(t) for t in terms[1:]) + ")"
    return "(" + (" AND " if op == "and" else " OR ").join(wording(t) for t in terms) + ")"


def decision(rules, default, facts):
    return next((outcome for condition, outcome in rules if truth(condition, facts)), default)


def distribution(rules, default, probabilities):
    result = [0.0] * 3
    for facts in itertools.product((False, True), repeat=4):
        mass = 1.0
        for value, probability in zip(facts, probabilities):
            mass *= probability if value else 1 - probability
        result[decision(rules, default, facts)] += mass
    return result


def main():
    if ROOT.exists():
        raise FileExistsError("workflow corpus is immutable")
    ROOT.mkdir(parents=True)
    (ROOT / "images").mkdir()
    config_path = Path("configs/workflow-release-v12.json")
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    source = Source(
        id="veyra-workflow-v12",
        revision="sha256:" + digest(Path(__file__).read_bytes()),
        license="Apache-2.0",
        url="https://github.com/ken-jo/veyra",
    )
    records, latent = [], []

    def add(
        group,
        split,
        family,
        state,
        kind,
        meanings,
        probabilities,
        tags,
        origin=source,
        image_sha=None,
        suffix=None,
    ):
        keys = [str(i) for i in range(len(meanings))]
        if kind == "noul":
            question = {"type": "noul", "instructions": meanings[0]}
            target = {"false": 1 - probabilities[2], "true": probabilities[2]}
            critical = "true"
        elif kind == "score":
            question = {
                "type": "score",
                "instructions": (
                    "Apply the supplied rules and return the distribution "
                    "over the ordered outcomes."
                ),
                "criteria": meanings,
            }
            target = dict(zip(keys, probabilities))
            critical = "2"
        else:
            keys = ["candidate_" + stable(group + str(i))[:6] for i in range(len(meanings))]
            question = {
                "type": "choice",
                "instructions": (
                    "Which outcome follows from the evidence and any supplied rules? "
                    "Return probabilities when facts are uncertain."
                ),
                "criteria": dict(zip(keys, meanings)),
            }
            target = dict(zip(keys, probabilities))
            critical = keys[2] if len(keys) == 3 and "workflow" in family else ""
        if not critical and family in {f[0] for f in FAMILIES}:
            critical = keys[2]
        gold = (
            ["gold_key:" + max(target, key=target.get)] if max(target.values()) >= 1 - 1e-9 else []
        )
        records.append(
            TrainingRecord(
                id=group + ":" + (suffix or kind),
                group_id=group,
                split=split,
                family=family,
                language="en",
                request=DecisionRequest.model_validate(
                    {"state": state, "questions": {"decision": question}}
                ),
                targets={"decision": target},
                source=origin,
                tags=[*tags, *gold, "critical_key:" + critical],
                image_sha256=image_sha,
            )
        )

    for family_index, (name, family_split, fields, rules, default) in enumerate(FAMILIES):
        assert {
            decision(rules, default, f) for f in itertools.product((False, True), repeat=4)
        } == {0, 1, 2}
        sizes = {family_split: protocol["data"]["new_workflow_groups_per_family"][family_split]}
        if family_split != "test":
            sizes["calibration"] = 40
        for split, count in sizes.items():
            for index in range(count):
                group = f"wf12:{name}:{split}:{index:04}"
                rng = random.Random(int(stable(group)[:16], 16))
                possible = [
                    f
                    for f in itertools.product((False, True), repeat=4)
                    if decision(rules, default, f) == index % 3
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
                        f"{n + 1}. IF {wording(condition)} THEN {ACTION[outcome]}."
                        for n, (condition, outcome) in enumerate(rules)
                    )
                    + f"\nOtherwise {ACTION[default]}."
                )
                prefix = (
                    f"Workflow: {name.replace('_', ' ')}. Case reference {stable(group)[:10]}.\n"
                    + "\n".join(definitions)
                    + "\n"
                    + policy
                )
                full = (
                    prefix + "\nObserved measurements: " + json.dumps(dict(zip(fields, readings)))
                )
                onehot = [float(i == index % 3) for i in range(3)]
                for kind in ("choice", "score", "noul"):
                    meanings = (
                        [f"Outcome {action}." for action in ACTION]
                        if kind != "noul"
                        else ["Under the supplied rules, is HOLD the required outcome?"]
                    )
                    add(
                        group,
                        split,
                        name,
                        {"text": full},
                        kind,
                        meanings,
                        onehot,
                        [
                            "domain:workflow_new",
                            "view:complete",
                            "family_role:" + family_split,
                            "rule_depth:" + str(len(rules)),
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
                for attempt in range(2):
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
                uncertainty = prefix + "\nObserved measurements: " + json.dumps(observed)
                uncertainty += (
                    "\nFor this question, unknown conditions are independent Bernoulli variables "
                    "with these specified prior probabilities: "
                )
                uncertainty += (
                    json.dumps({"ABCD"[i]: priors[i] for i in sorted(hidden)})
                    + ". All unreported information remains unknown."
                )
                if condition == "conflicting":
                    uncertainty += (
                        f"\nTwo independent sensors report on condition {'ABCD'[conflict]}. "
                        "Sensor X says TRUE and is correct with probability 0.80. "
                        "Sensor Y says FALSE and is correct with probability 0.65. "
                        "Each sensor has equal sensitivity and specificity. "
                        "Their errors are independent conditional on the true condition. "
                        "Use the reports and prior together."
                    )
                meanings = (
                    [f"Outcome {action}." for action in ACTION]
                    if kind != "noul"
                    else [
                        "Under the supplied rules and stated uncertainty, "
                        "what is the probability that HOLD is required?"
                    ]
                )
                add(
                    group,
                    split,
                    name,
                    {"text": uncertainty},
                    kind,
                    meanings,
                    target,
                    [
                        "domain:uncertainty",
                        "view:partial_evidence",
                        "known_conditional_distribution",
                        "condition:" + condition,
                        "family_role:" + family_split,
                    ],
                    suffix="uncertain",
                )
                latent.append(
                    {
                        "group": group,
                        "condition": condition,
                        "posterior": posterior,
                        "outcome_distribution": target,
                    }
                )

    # Preserve previous source split boundaries, including previously inspected holdouts.
    for record in read_records(Path("data/transfer-v10/records.jsonl")):
        if record.source.id != "LocalLLaMA/typed-decisions" or record.split == "test":
            continue
        item = record.model_copy(deep=True)
        item.tags = [t for t in item.tags if not t.startswith("domain:")]
        item.tags += [
            "domain:workflow_known",
            "teacher_distribution",
            "legacy_development" if item.split == "dev" else "upstream_train",
        ]
        records.append(item)

    # Replay only the frozen baseline's training observations, never its final groups.
    replay = defaultdict(lambda: defaultdict(list))
    for record in read_records(Path("data/foundation-v11/records.jsonl")):
        if record.split == "train":
            domain = next(t[7:] for t in record.tags if t.startswith("domain:"))
            replay[domain][record.group_id].append(record)
    quotas = {
        "image_waste": 300,
        "image_leaf": 100,
        "text_nli": 300,
        "text_intent": 300,
        "retention": 80,
    }
    for domain, limit in quotas.items():
        for group in sorted(replay[domain], key=stable)[:limit]:
            for record in replay[domain][group]:
                item = record.model_copy(deep=True)
                item.tags += ["retention_replay"]
                for image in item.request.state.images:
                    origin = Path("data/foundation-v11") / image.path
                    destination = ROOT / "images" / origin.name
                    if not destination.exists():
                        os.link(origin, destination)
                    image.path = str(destination.relative_to(ROOT)).replace("\\", "/")
                records.append(item)

    selected = json.loads(
        Path("data/workflow-v12-source/retention-selections.json").read_text(encoding="utf-8")
    )
    for row in selected["text_rows"]:
        group = "fresh12:" + row["source"] + ":" + row["id"]
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
                upstream_split=row["upstream_split"],
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
                upstream_split=row["upstream_split"],
            )
        add(
            group,
            row["split"],
            domain,
            {"text": row["text"]},
            "choice",
            meanings,
            [float(i == label) for i in range(len(meanings))],
            ["domain:" + domain, "view:recognition", "fresh_retention"],
            origin,
        )

    download = json.loads(
        Path("data/workflow-v12-source/downloads.json").read_text(encoding="utf-8")
    )
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
    signatures, counts, excluded = [], Counter(), 0
    for row in sorted(selected["image_candidates"], key=lambda r: stable(r["sha256"])):
        label = row["label"]
        if counts[label] >= 150:
            continue
        raw = Path(row["path"]).read_bytes()
        pixel, phash, dhash, color = image_signature(raw)
        duplicate = any(
            pixel == p
            or (
                (phash ^ ph).bit_count() <= 4
                and (dhash ^ dh).bit_count() <= 8
                and max(abs(a - b) for a, b in zip(color, c)) <= 35
            )
            for p, ph, dh, c in signatures
        )
        if duplicate:
            excluded += 1
            continue
        signatures.append((pixel, phash, dhash, color))
        n = counts[label]
        split = "dev" if n < 60 else "calibration" if n < 90 else "test"
        counts[label] += 1
        path = ROOT / "images" / (row["sha256"] + ".png")
        os.link(Path(row["path"]), path)
        group = "fresh12:cifar:" + row["sha256"][:24]
        add(
            group,
            split,
            "photo_guard",
            {
                "text": "Identify the main object in the photograph.",
                "images": [{"path": "images/" + path.name}],
            },
            "choice",
            ["The main object is a " + x + "." for x in labels],
            [float(i == label) for i in range(10)],
            ["domain:photo_guard", "view:recognition", "fresh_retention", "evaluation_only"],
            origin,
            digest(raw),
        )
    if any(counts[i] != 150 for i in range(10)):
        raise ValueError("insufficient distinct photographs: " + str(counts))
    summary = audit_records(records)
    path = ROOT / "records.jsonl"
    path.write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8", newline="\n"
    )
    latent_path = ROOT / "conditional-target-audit.json"
    latent_path.write_text(json.dumps(latent, indent=2) + "\n", encoding="utf-8")
    result = {
        **summary,
        "records_sha256": digest(path.read_bytes()),
        "release_protocol_sha256": digest(config_path.read_bytes()),
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "family_split": {f[0]: f[1] for f in FAMILIES},
        "cifar_similar_images_excluded": excluded,
        "domain_split_questions": {},
        "final_inference_performed": False,
        "latent_targets_not_used_at_inference": True,
        "conditional_target_audit_sha256": digest(latent_path.read_bytes()),
    }
    for domain in sorted({next(t[7:] for t in r.tags if t.startswith("domain:")) for r in records}):
        result["domain_split_questions"][domain] = dict(
            Counter(r.split for r in records if "domain:" + domain in r.tags)
        )
    (ROOT / "protocol.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
