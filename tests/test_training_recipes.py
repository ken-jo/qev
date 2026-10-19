import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import option_mix_common as common  # noqa: E402
import train_option_mix as mix  # noqa: E402


@pytest.mark.parametrize("script", ["train_option_mix.py", "train_exit_head.py"])
def test_recipe_help_runs_without_loading_a_model(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "--base" in result.stdout
    assert "--items" in result.stdout


def item(identifier, options=("red", "blue", "green"), gold="B", source="colors"):
    return {
        "id": identifier,
        "source": source,
        "question": "Which color is the square?",
        "options": list(options),
        "gold": gold,
        "images": ["square.png"],
    }


def test_items_split_stably_and_resolve_images_next_to_the_file(tmp_path):
    folder = tmp_path / "colors"
    folder.mkdir()
    rows = [item(f"item-{index}") for index in range(400)]
    (folder / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    train, dev = common.load_items([folder / "items.jsonl"], dev_per_source=5)
    again_train, again_dev = common.load_items([folder / "items.jsonl"], dev_per_source=5)
    assert len(dev) == 5 and len(train) == 395
    assert [e["item"]["id"] for e in dev] == [e["item"]["id"] for e in again_dev]
    request = train[0]["request"]
    assert request.state.images[0].path == str((folder / "square.png").resolve())
    assert request.questions["answer"].criteria == {"A": "red", "B": "blue", "C": "green"}
    capped, capped_dev = common.load_items([folder / "items.jsonl"], caps={"colors": 40})
    assert len(capped) + len(capped_dev) == 40


def test_duplicate_or_blank_options_stay_distinct():
    request = common.item_request(item("x", options=("same", "same", " ")))
    assert request.questions["answer"].criteria == {"A": "same", "B": "same (B)", "C": "C"}
    with pytest.raises(ValueError, match="gold"):
        common.item_example(item("x", gold="D"))


def test_permutation_moves_gold_with_its_option_content():
    example = common.item_example(item("x", options=("a", "b", "c", "d", "e"), gold="D"))
    for seed in range(10):
        moved = common.permute(example, random.Random(seed))
        criteria = moved["request"].questions["answer"].criteria
        assert criteria[moved["gold"]["answer"]] == "d"
        assert sorted(criteria.values()) == ["a", "b", "c", "d", "e"]
    fixed = common.item_example(item("x", source="options_in_image"))
    assert common.permute(fixed, random.Random(0)) is fixed


def test_losses_follow_gold_or_soft_targets():
    example = common.item_example(item("x"))
    logits = {"answer": torch.tensor([0.0, 5.0, 0.0])}
    assert common.example_loss(logits, example) < common.example_loss(
        {"answer": torch.tensor([5.0, 0.0, 0.0])}, example
    )
    soft = dict(example, targets={"answer": {"A": 0.5, "B": 0.5, "C": 0.0}})
    expected = -(torch.tensor([0.5, 0.5, 0.0]) * logits["answer"].log_softmax(-1)).sum()
    assert torch.isclose(common.example_loss(logits, soft), expected)


def test_multi_image_requests_share_the_total_pixel_budget():
    processor = SimpleNamespace(size={"shortest_edge": 65536, "longest_edge": 262144})
    model = SimpleNamespace(
        encoder=SimpleNamespace(processor=SimpleNamespace(image_processor=processor))
    )
    common.set_pixel_budget(model, 1638400, 1, 2097152)
    assert processor.size == {"shortest_edge": 65536, "longest_edge": 1638400}
    common.set_pixel_budget(model, 1638400, 4, 2097152)
    assert processor.size["longest_edge"] == 524288
    common.set_pixel_budget(model, 1638400, 4, 131072)
    assert processor.size["longest_edge"] == 65536


@pytest.mark.parametrize("failures", [False, True])
def test_mix_evaluation_retains_failed_questions_and_sources(failures):
    class FixedModel:
        encoder = SimpleNamespace(
            processor=SimpleNamespace(
                image_processor=SimpleNamespace(size={"shortest_edge": 65536})
            )
        )

        def eval(self):
            pass

        def __call__(self, request, root):
            if failures and root == "fail":
                raise ValueError("invalid request")
            return {name: torch.tensor([0.0, 5.0, 0.0]) for name in request.questions}, 0

    example = common.item_example(item("ok"))
    examples = [
        dict(example, root="ok"),
        dict(example, root="fail"),
        dict(example, root="fail", kind="other"),
    ]
    result = mix.evaluate(FixedModel(), examples, [262144], 2097152)
    kind = example["kind"]
    assert result[kind] == {
        "accuracy": 0.5 if failures else 1.0,
        "n": 2,
        "completed": 1 if failures else 2,
        "failed": 1 if failures else 0,
    }
    assert result["other"] == {
        "accuracy": 0.0 if failures else 1.0,
        "n": 1,
        "completed": 0 if failures else 1,
        "failed": 1 if failures else 0,
    }
    assert result["select"] == (0.25 if failures else 1.0)


def test_exit_distillation_learns_a_separable_teacher():
    import runpy

    recipe = runpy.run_path(str(ROOT / "scripts" / "train_exit_head.py"))
    torch.manual_seed(0)
    rows = []
    for index in range(96):
        label = index % 3
        features = torch.randn(1, 16) * 0.1
        features[0, label] += 3.0
        teacher = torch.full((3,), -2.0)
        teacher[label] = 2.0
        rows.append(
            {
                "features": features,
                "positions": [0, 1, 2],
                "teacher": teacher,
                "gold": label,
                "kind": "item-toy",
            }
        )
    args = SimpleNamespace(
        device="cpu", learning_rate=1e-2, epochs=8, batch_size=32, temperature=1.0, gold_weight=0.3
    )
    head = recipe["distill"](rows, 0, 8, args)
    result = recipe["score"](head, rows, 0, "cpu")["item-toy"]
    assert result["n"] == 96
    assert result["agreement"] == 1.0
    assert result["exit"] == result["full"] == 1.0
