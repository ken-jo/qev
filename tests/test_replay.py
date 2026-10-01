import random

import pytest

from veyra.build_data import text_records
from veyra.replay import audit_replay, replay_pool, sample_replay


def test_replay_has_only_single_questions_and_preserves_targets():
    original = [r for r in text_records(12, 81) if r.split == "train"]
    pool = replay_pool(original)
    selected = sample_replay(pool, 17, random.Random(3))
    originals = {r.id: r for r in original}
    assert len(selected) == 17
    for record in selected:
        assert record.split == "train"
        assert len(record.request.questions) == 1
        name = next(iter(record.request.questions))
        assert record.targets[name] == originals[record.id].targets[name]
        assert record.request.state == originals[record.id].request.state
    assert selected == sample_replay(pool, 17, random.Random(3))


def test_replay_rejects_held_out_records():
    for record in text_records(20, 81):
        if record.split != "train":
            with pytest.raises(ValueError, match="training"):
                replay_pool([record])


def test_replay_context_copies_do_not_multiply_sampling_weight():
    records = [r for r in text_records(12, 81) if r.split == "train"]
    duplicate = records[0].model_copy(deep=True)
    duplicate.id += "-context"
    before = replay_pool(records)
    after = replay_pool(records + [duplicate])
    assert {k: len(v) for k, v in before.items()} == {k: len(v) for k, v in after.items()}


def test_replay_cannot_reintroduce_a_held_out_question(tmp_path):
    source = next(r for r in text_records(12, 81) if r.split == "train")
    heldout = source.model_copy(deep=True)
    heldout.split = "test"
    heldout.id = "held-out"
    heldout.group_id = "held-out"
    with pytest.raises(ValueError, match="question overlaps"):
        audit_replay([heldout], tmp_path, [source], tmp_path)
