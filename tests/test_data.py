import pytest

from veyra.data import TrainingRecord, audit_records, read_records, write_records


def record(name="r", group="g", split="train"):
    return TrainingRecord.model_validate(
        {
            "id": name,
            "group_id": group,
            "split": split,
            "family": "test",
            "language": "en",
            "request": {
                "state": {"text": "hello"},
                "questions": {"q": {"type": "noul", "instructions": "Greeting?"}},
            },
            "targets": {"q": {"true": 1.0, "false": 0.0}},
            "source": {
                "id": "original",
                "revision": "1",
                "license": "Apache-2.0",
                "url": "https://github.com/ken-jo/veyra",
            },
        }
    )


def test_group_and_image_split_leakage():
    with pytest.raises(ValueError, match="group crosses"):
        audit_records([record(), record("r2", split="test")])
    a, b = record(), record("r2", "g2", "test")
    a.image_sha256 = b.image_sha256 = "a" * 64
    with pytest.raises(ValueError, match="image bytes"):
        audit_records([a, b])


def test_distribution_validation():
    data = record().model_dump()
    data["targets"]["q"] = {"false": 0.1, "true": 0.1}
    with pytest.raises(ValueError, match="sum to one"):
        TrainingRecord.model_validate(data)


def test_round_trip(tmp_path):
    records = [record(), record("r2", "g2", "test")]
    path = tmp_path / "data.jsonl"
    assert write_records(path, records)["split_counts"]["test"] == 1
    assert read_records(path) == records
