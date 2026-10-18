import pytest
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from veyra.image_inputs import open_images
from veyra.option_model import MAX_IMAGE_PIXELS, OptionConfig
from veyra.schema import DecisionRequest

CAP = 262144


def images(*paths):
    return DecisionRequest.model_validate(
        {
            "state": {"images": [{"path": str(path)} for path in paths]},
            "questions": {
                "q": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "a", "b": "b"}}
            },
        }
    ).state.images


def test_option_config_accepts_the_larger_resolution_ceiling_only_on_request():
    assert OptionConfig().image_max_pixels == 262144
    assert OptionConfig(image_max_pixels=MAX_IMAGE_PIXELS).image_max_pixels == 1638400
    with pytest.raises(ValueError, match="pixel budget"):
        OptionConfig(image_max_pixels=MAX_IMAGE_PIXELS + 1)


def test_large_jpeg_decodes_at_a_reduced_scale_that_still_covers_the_budget(tmp_path):
    path = tmp_path / "large.jpg"
    Image.new("RGB", (2048, 1536), (90, 140, 200)).save(path, quality=90)
    with open_images(images(path), max_pixels=CAP) as (image,):
        assert image.mode == "RGB"
        assert image.size == (1024, 768)
        assert image.width * image.height >= CAP
        drafted = smart_resize(image.height, image.width, 32, 65536, CAP)
    with open_images(images(path)) as (image,):
        assert image.size == (2048, 1536)
        full = smart_resize(image.height, image.width, 32, 65536, CAP)
    # The processor resize, not the draft, still decides the encoded image grid.
    assert drafted == full


def test_small_jpegs_and_other_formats_decode_at_full_resolution(tmp_path):
    small, lossless = tmp_path / "small.jpg", tmp_path / "large.png"
    Image.new("RGB", (700, 700), "white").save(small)
    Image.new("RGB", (2048, 1536), "white").save(lossless)
    with open_images(images(small, lossless), max_pixels=CAP) as (first, second):
        assert first.size == (700, 700)
        assert second.size == (2048, 1536)


def test_decode_limit_is_checked_before_any_draft(tmp_path):
    path = tmp_path / "huge.jpg"
    Image.new("L", (8000, 5001)).save(path, quality=10)
    with pytest.raises(ValueError, match="40 megapixel"):
        with open_images(images(path), max_pixels=CAP):
            pass
