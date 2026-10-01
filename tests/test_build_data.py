from veyra.build_data import build_dataset
from veyra.data import read_records


def test_original_generator_is_grounded_and_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build_dataset(a, scenes=24, text_cases=24, seed=19, include_beans=False)
    build_dataset(b, scenes=24, text_cases=24, seed=19, include_beans=False)
    first, second = read_records(a / "records.jsonl"), read_records(b / "records.jsonl")
    assert first == second
    assert any(record.image_sha256 for record in first)
    assert any("missing_evidence" in record.tags for record in first)
    assert any("image_text_conflict" in record.tags for record in first)
    assert set(record.language for record in first) == {"en", "ko"}
    assert {
        question.type for record in first for question in record.request.questions.values()
    } == {"choice", "score", "noul"}


def test_modified_image_fails_integrity_audit(tmp_path):
    import pytest

    build_dataset(tmp_path, scenes=1, text_cases=0, seed=19, include_beans=False)
    image = next((tmp_path / "images").rglob("*.png"))
    image.write_bytes(b"not the original image")
    with pytest.raises(ValueError, match="checksum"):
        read_records(tmp_path / "records.jsonl")
