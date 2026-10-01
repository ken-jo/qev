import random

import pytest

from veyra.build_data import make_record
from veyra.interventions import augment_unit, primitive_question, retype_unit, training_units
from veyra.policy_data import typed_decision


def record(kind="choice", split="train", version=0):
    labels = ["dispatch", "hold"]
    outcomes = labels[version:] + labels[:version]
    question = typed_decision(
        random.Random(1),
        f"If weight >= 10, return '{outcomes[0]}'. Otherwise return '{outcomes[1]}'.",
        labels,
        outcomes[0],
        kind,
        "dispatch",
    )
    return make_record(
        f"observation-rule-{version}",
        "observation",
        split,
        "text_threshold",
        "en",
        {"text": "Observation: weight = 20; items = 4."},
        {"q": question},
        {"id": "unit", "revision": "unit", "license": "Apache-2.0", "url": "https://example.com"},
    )


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_primitive_auxiliary_has_condition_truth(kind):
    auxiliary = primitive_question(record(kind))
    assert auxiliary.targets["condition"]["true"] == 1
    assert "weight >= 10" in auxiliary.request.questions["condition"].instructions
    assert "dispatch" not in auxiliary.request.questions["condition"].instructions


def test_joint_intervention_changes_symbols_without_changing_targets():
    original = [record(version=0), record(version=1)]
    changed = augment_unit(original, random.Random(40))
    for before, after in zip(original, changed, strict=True):
        assert before.targets == after.targets
        assert "weight" not in after.request.state.text
        assert "dispatch" not in after.request.questions["q"].instructions
    assert changed[0].request.state.text == changed[1].request.state.text
    units = training_units(original)
    assert [len(unit) for unit in units] == [2, 1]


def test_held_out_records_cannot_be_transformed():
    for split in ("dev", "calibration", "test"):
        heldout = record(split=split)
        with pytest.raises(ValueError, match="training|held-out"):
            primitive_question(heldout)
        with pytest.raises(ValueError, match="training|held-out"):
            augment_unit([heldout], random.Random(1))
        with pytest.raises(ValueError, match="training|held-out"):
            training_units([heldout])


@pytest.mark.parametrize("desired", [0, 1])
def test_evidence_values_really_change_the_condition(desired):
    import re

    from veyra.evidence_data import text_observation

    for seed in range(30):
        text = text_observation(record(), desired, random.Random(seed))
        weight = int(re.search(r"weight = (\d+)", text).group(1))
        assert (weight >= 10) == (desired == 0)


def test_condition_annotation_can_supervise_false_noul_probe():
    source = record(kind="noul", version=1)
    source.tags.append("oracle_condition_branch:0")
    derived = primitive_question(source)
    assert derived.targets["condition"]["true"] == 1


@pytest.mark.parametrize("source_kind", ["choice", "score", "noul"])
def test_type_interventions_preserve_policy_outcomes(source_kind):
    import re

    from veyra.candidates import candidates_for

    seen = set()
    original = [record(source_kind, version=0), record(source_kind, version=1)]
    for seed in range(50):
        changed = retype_unit(original, random.Random(seed))
        for before, after in zip(original, changed, strict=True):
            question = after.request.questions["q"]
            seen.add(question.type)
            expected = re.search(r"return '([^']+)'", question.instructions).group(1)
            winner = max(candidates_for(question), key=lambda c: after.targets["q"][c.key])
            if question.type == "noul":
                probe = re.search(
                    r"Is the policy outcome '([^']+)'\?", question.instructions
                ).group(1)
                assert after.targets["q"]["true"] == float(expected == probe)
            else:
                assert winner.description == expected
            assert before.request.state == after.request.state
    assert seen == {"choice", "score", "noul"}


def test_type_interventions_reject_held_out_records():
    with pytest.raises(ValueError, match="training"):
        retype_unit([record(split="dev", version=i) for i in (0, 1)], random.Random(1))


@pytest.mark.parametrize("branch", [1, 2])
def test_type_interventions_follow_all_three_range_branches(branch):
    import re

    from veyra.candidates import candidates_for

    originals = []
    labels = ["low", "middle", "high"]
    for version in (0, 1):
        outcomes = labels[version:] + labels[:version]
        instructions = (
            f"If weight < 10, return '{outcomes[0]}'. "
            f"If 10 <= weight < 20, return '{outcomes[1]}'. "
            f"If weight >= 20, return '{outcomes[2]}'."
        )
        question, target = typed_decision(
            random.Random(1), instructions, labels, outcomes[branch], "noul", "high"
        )
        raw = record("noul", version=version).model_dump()
        raw["request"]["questions"] = {"q": question}
        raw["targets"] = {"q": target}
        raw["tags"] = [f"oracle_condition_branch:{branch}"]
        raw["request"]["state"]["text"] = f"Observation: weight = {15 if branch == 1 else 25}."
        originals.append(type(record()).model_validate(raw))
    for seed in range(40):
        for converted in retype_unit(originals, random.Random(seed)):
            question = converted.request.questions["q"]
            expected = re.findall(r"return '([^']+)'", question.instructions)[branch]
            if question.type == "noul":
                probe = re.search(
                    r"Is the policy outcome '([^']+)'\?", question.instructions
                ).group(1)
                assert converted.targets["q"]["true"] == float(probe == expected)
            else:
                winner = max(candidates_for(question), key=lambda c: converted.targets["q"][c.key])
                assert winner.description == expected
