from veyra.data import TrainingRecord
from veyra.policy_refresh import LABELS_V8, WRAPPERS_V8, remap_record


def test_fresh_final_remaps_fields_outcomes_and_proposition_without_changing_targets():
    record = TrainingRecord.model_validate(
        {
            "id": "fresh-r0",
            "group_id": "fresh",
            "split": "test",
            "family": "text_threshold",
            "language": "en",
            "request": {
                "state": {"text": "Observation: sample index = 23; buffer depth = 8."},
                "questions": {
                    "q": {
                        "type": "noul",
                        "instructions": "Old wrapper. If sample index >= 20, return 'Lumen desk'. "
                        "Otherwise return 'Quartz desk'. Is the policy outcome 'Lumen desk'?",
                    }
                },
            },
            "targets": {"q": {"false": 0.0, "true": 1.0}},
            "source": {"id": "old", "revision": "fixed", "license": "Apache-2.0", "url": "local"},
        }
    )
    refreshed = remap_record(record, "raw", "sha256:new")
    assert refreshed.targets == record.targets
    assert refreshed.request.state.text == "Observation: crystal reading = 23; turbine rating = 8."
    question = refreshed.request.questions["q"]
    assert question.instructions.startswith(WRAPPERS_V8["test"])
    assert "If crystal reading >= 20" in question.instructions
    assert f"Is the policy outcome '{LABELS_V8['test'][0]}'?" in question.instructions
    assert "Lumen desk" not in question.instructions
    assert record.request.state.text.startswith("Observation: sample index")
