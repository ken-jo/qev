import pytest
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from veyra.image_inputs import open_images
from veyra.option_model import MAX_IMAGE_PIXELS, OptionConfig
from veyra.schema import DecisionRequest

CAP = 262144


def images(*paths):
    return request(*paths).state.images


def request(*paths):
    return DecisionRequest.model_validate(
        {
            "state": {"text": "Evidence.", "images": [{"path": str(path)} for path in paths]},
            "questions": {
                "q": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "a", "b": "b"}}
            },
        }
    )


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


class StopEncoding(Exception):
    pass


def captured_budget(monkeypatch, module, call):
    """Run an encoder up to image decoding and report the draft budget it requested."""
    seen = {}

    def fake_open_images(images, image_root=None, *, max_pixels=None):
        seen["max_pixels"] = max_pixels
        raise StopEncoding

    monkeypatch.setattr(module, "open_images", fake_open_images)
    with pytest.raises(StopEncoding):
        call()
    return seen["max_pixels"]


def encoders(config):
    from types import SimpleNamespace

    from veyra.backbone import QwenEncoder

    processor = SimpleNamespace(
        image_processor=SimpleNamespace(size={"shortest_edge": 65536, "longest_edge": 524288})
    )
    return QwenEncoder(None, processor, config)


@pytest.mark.parametrize("enabled", [False, True])
def test_fast_jpeg_decode_is_opt_in_for_the_option_encoder(monkeypatch, enabled):
    import veyra.option_model as option_model

    model = option_model.OptionModel.__new__(option_model.OptionModel)
    option_model.nn.Module.__init__(model)
    model.encoder = encoders(OptionConfig(fast_jpeg_decode=enabled))
    budget = captured_budget(monkeypatch, option_model, lambda: model.encode(request()))
    assert budget == (524288 if enabled else None)


@pytest.mark.parametrize("enabled", [False, True])
def test_fast_jpeg_decode_is_opt_in_for_the_packed_encoder(monkeypatch, enabled):
    import veyra.backbone as backbone

    encoder = encoders(backbone.EncoderConfig(fast_jpeg_decode=enabled))
    budget = captured_budget(monkeypatch, backbone, lambda: encoder(request()))
    assert budget == (524288 if enabled else None)


def test_fast_jpeg_decode_flag_only_appears_in_manifests_when_enabled():
    from veyra.backbone import EncoderConfig

    for config_type in (OptionConfig, EncoderConfig):
        assert "fast_jpeg_decode" not in config_type().to_dict()
        enabled = config_type(fast_jpeg_decode=True)
        assert enabled.to_dict()["fast_jpeg_decode"] is True
        assert config_type(**enabled.to_dict()) == enabled
        with pytest.raises(ValueError, match="boolean"):
            config_type(fast_jpeg_decode=1)
