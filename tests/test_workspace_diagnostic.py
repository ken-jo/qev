import random
import runpy
from pathlib import Path

import pytest

from veyra.build_data import make_record
from veyra.policy_data import typed_decision

development_conditions = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_workspace_conditions.py")
)["development_conditions"]


def pair(kind, branch, split="dev"):
    labels = ["amber", "blue", "copper"]
    records = []
    for version in (0, 1):
        outcomes = labels[version:] + labels[:version]
        instructions = (
            f"If load < 10, return '{outcomes[0]}'. "
            f"If 10 <= load < 20, return '{outcomes[1]}'. "
            f"If load >= 20, return '{outcomes[2]}'."
        )
        record = make_record(
            f"observation-rule-{version}",
            "observation",
            split,
            "text_ranges",
            "en",
            {"text": f"Observation: load = {5 + 10 * branch}."},
            {
                "q": typed_decision(
                    random.Random(3), instructions, labels, outcomes[branch], kind, labels[branch]
                )
            },
            {"id": "unit", "revision": "unit", "license": "Apache-2.0", "url": "local"},
        )
        record.tags.append("counterfactual_pair")
        records.append(record)
    return records


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
@pytest.mark.parametrize("branch", [0, 1, 2])
def test_diagnostic_uses_observation_branch_across_output_changes(kind, branch):
    records = pair(kind, branch)
    before = [record.model_dump_json() for record in records]
    assert development_conditions(list(reversed(records))) == {"observation": branch}
    assert [record.model_dump_json() for record in records] == before


@pytest.mark.parametrize("split", ["train", "calibration", "test"])
def test_diagnostic_rejects_non_development_annotations(split):
    with pytest.raises(ValueError, match="development"):
        development_conditions(pair("choice", 0, split))


def test_diagnostic_rejects_incomplete_counterfactuals():
    with pytest.raises(ValueError, match="two one-question"):
        development_conditions(pair("noul", 1)[:1])
