"""Validate the public demo's request adapter without loading the backbone."""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("space_engine", ROOT / "apps/hf_space/engine.py")
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)


def test_english_presets_produce_valid_requests():
    presets = json.loads((ROOT / "apps/hf_space/presets.json").read_text("utf-8"))
    count = 0
    for preset in presets.values():
        for question in preset["questions"]:
            criteria = "\n".join(
                item["text"] if question["type"] == "score" else item["key"] + " | " + item["text"]
                for item in question["options"]
            )
            request = engine.build_request(
                preset["text"],
                question["instructions"],
                question["type"],
                criteria,
                "sampleId" in preset,
            )
            assert len(request.questions) == 1
            assert "sample-" not in request.model_dump_json()
            count += 1
    assert count == 10


@pytest.mark.parametrize(
    "kind,criteria",
    [
        ("choice", "x | First\nx | Second"),
        ("choice", "Missing separator\ny | Second"),
        ("noul", "yes | Yes\nno | No"),
        ("noul", "false | False\ntrue | True\nextra | Extra"),
        ("score", "Only one level"),
        ("other", "x | First\ny | Second"),
    ],
)
def test_invalid_candidates_are_rejected(kind, criteria):
    with pytest.raises(ValueError):
        engine.build_request("Evidence", "Question?", kind, criteria, False)


def test_image_resize_preserves_shape_and_does_not_upscale():
    image = Image.new("RGB", (512, 384), "white")
    resized, _ = engine.prepare_image(image, "64")
    assert resized.size == (64, 48)
    resized.close()
    unchanged, _ = engine.prepare_image(image, "1024")
    assert unchanged.size == image.size
    unchanged.close()
    image.close()


def test_unrecognized_resolution_is_rejected():
    with Image.new("RGB", (32, 32)) as image, pytest.raises(ValueError):
        engine.prepare_image(image, "999999")


def test_no_model_means_no_fake_prediction():
    with pytest.raises(RuntimeError, match="not ready"):
        engine.DemoEngine().predict("text", None, "Question?", "choice", "x | X\ny | Y")
