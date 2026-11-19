"""Opt-in fast image path: array input, device preprocessing, vision graphs and budgets."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from test_option_model import fake_exit_backbone, mode_checkpoint

from qev_runtime.image_inputs import device_images, open_images, pixel_fingerprints
from qev_runtime.option_model import OptionConfig, OptionModel
from qev_runtime.schema import ArrayImageInput, DecisionRequest, ImageInput
from qev_runtime.vision_graph import VisionGraphs, enable_vision_graphs

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
BUDGETS = [(65536, 1638400), (65536, 262144), (1024, 50176), (1024, 12544)]
SIZES = [(1080, 1920), (500, 37), (200, 3000), (480, 640), (33, 33)]


def request(*images):
    return DecisionRequest.model_validate(
        {
            "state": {"text": "Evidence.", "images": list(images)},
            "questions": {
                "q": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "a", "b": "b"}}
            },
        }
    )


def pattern(height, width, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def processor():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    # The Qwen3.5 vision settings: 16-pixel patches, 2x2 merge, mean/std 0.5.
    return Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, temporal_patch_size=2, image_mean=0.5, image_std=0.5
    )


# Array input


def test_numpy_and_torch_uint8_arrays_are_image_evidence_in_order(tmp_path):
    path = tmp_path / "a.png"
    Image.fromarray(pattern(4, 5)).save(path)
    array, tensor = pattern(6, 7, 1), torch.from_numpy(pattern(8, 9, 2))
    parsed = request({"pixels": array}, {"path": str(path)}, ArrayImageInput(pixels=tensor))
    kinds = [type(item) for item in parsed.state.images]
    assert kinds == [ArrayImageInput, ImageInput, ArrayImageInput]
    assert parsed.state.images[0].pixels is array
    with open_images(parsed.state.images) as (first, second, third):
        assert first.size == (7, 6) and second.size == (5, 4) and third.size == (9, 8)
        assert np.array_equal(np.asarray(first), array)
        assert np.array_equal(np.asarray(third), tensor.numpy())


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((4, 4, 3), dtype=np.float32),
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((3, 4, 4), dtype=np.uint8),
        np.zeros((0, 4, 3), dtype=np.uint8),
        torch.zeros(4, 4, 3),
        [[[0, 0, 0]]],
        "pixels",
    ],
)
def test_arrays_must_be_hwc_rgb_uint8(pixels):
    with pytest.raises(ValueError):
        request({"pixels": pixels})


def test_arrays_share_the_40_megapixel_limit():
    huge = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (8000, 5001, 3))
    with pytest.raises(ValueError, match="40 megapixel"):
        request({"pixels": huge})


def test_path_requests_keep_their_json_schema_and_serialisation():
    schema = DecisionRequest.model_json_schema()
    assert schema["$defs"]["State"]["properties"]["images"]["items"] == {
        "$ref": "#/$defs/ImageInput"
    }
    assert "pixels" not in json.dumps(schema) and "ArrayImageInput" not in json.dumps(schema)
    parsed = request({"path": "screen.png"})
    assert parsed.model_dump()["state"]["images"] == [{"path": "screen.png"}]
    assert '"images":[{"path":"screen.png"}]' in parsed.model_dump_json()
    with pytest.raises(ValueError):
        DecisionRequest.from_json(
            json.dumps({"state": {"images": [{"pixels": [[[0, 0, 0]]]}]}, "questions": {}})
        )
    described = request({"pixels": pattern(2, 3)}).model_dump()
    assert described["state"]["images"] == [{"pixels": "uint8[2,3,3]"}]


def test_array_fingerprints_match_a_lossless_file_of_the_same_pixels(tmp_path):
    array = pattern(10, 12)
    Image.fromarray(array).save(tmp_path / "same.png")
    from_file = pixel_fingerprints(request({"path": "same.png"}).state.images, tmp_path, {})
    from_array = pixel_fingerprints(request({"pixels": array}).state.images, tmp_path, {})
    assert from_file == from_array
    expected = hashlib.sha256(str((12, 10)).encode() + array.tobytes()).hexdigest()
    assert from_array == [expected]


def test_device_images_are_the_decoded_pixels_as_chw_tensors(tmp_path):
    array = pattern(9, 11)
    Image.fromarray(array).save(tmp_path / "x.png")
    images = request({"path": "x.png"}, {"pixels": array}).state.images
    from_file, from_array = device_images(images, tmp_path, "cpu")
    for tensor in (from_file, from_array):
        assert tensor.dtype == torch.uint8 and tensor.shape == (3, 9, 11)
        assert torch.equal(tensor, torch.from_numpy(array).permute(2, 0, 1))


# Grid computation


@pytest.mark.parametrize("minimum,maximum", BUDGETS)
def test_tensor_preprocessing_matches_the_pil_path_grid_and_pixels(minimum, maximum):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    ip = processor()
    ip.size = {"shortest_edge": minimum, "longest_edge": maximum}
    for height, width in SIZES:
        array = pattern(height, width)
        stock = ip(images=[Image.fromarray(array)], return_tensors="pt")
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        fast = ip(images=[tensor], return_tensors="pt", device="cpu")
        assert torch.equal(fast["image_grid_thw"], stock["image_grid_thw"])
        assert torch.equal(fast["pixel_values"], stock["pixel_values"])
        h, w = smart_resize(height, width, 32, minimum, maximum)
        assert stock["image_grid_thw"].tolist() == [[1, h // 16, w // 16]]


# Flag plumbing


def test_device_preprocess_is_an_opt_in_manifest_flag():
    assert OptionConfig().device_preprocess is False
    assert "device_preprocess" not in OptionConfig().to_dict()
    enabled = OptionConfig(device_preprocess=True)
    assert enabled.to_dict()["device_preprocess"] is True
    assert OptionConfig(**enabled.to_dict()) == enabled
    with pytest.raises(ValueError, match="boolean"):
        OptionConfig(device_preprocess=1)


class StopEncoding(Exception):
    pass


def encoding_calls(monkeypatch, config, device):
    """Run ``encode`` up to the processor and report which image path it took."""
    import qev_runtime.option_model as option_model

    calls = {}

    def fake_open_images(images, image_root=None, *, max_pixels=None):
        calls["open_images"] = max_pixels
        raise StopEncoding

    def fake_device_images(images, image_root, target, *, max_pixels=None):
        calls["device_images"] = (str(target), max_pixels)
        return [torch.zeros(3, 4, 4, dtype=torch.uint8)]

    class FakeProcessor:
        image_processor = SimpleNamespace(size={"shortest_edge": 1024, "longest_edge": 50176})

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            calls["processor_device"] = kwargs.get("device")
            raise StopEncoding

    monkeypatch.setattr(option_model, "open_images", fake_open_images)
    monkeypatch.setattr(option_model, "device_images", fake_device_images)
    model = option_model.OptionModel.__new__(option_model.OptionModel)
    option_model.nn.Module.__init__(model)
    model.encoder = SimpleNamespace(
        config=config, processor=FakeProcessor(), device=torch.device(device)
    )
    with pytest.raises(StopEncoding):
        model.encode(request({"path": "unused.png"}))
    return model, calls


def test_device_preprocess_routes_images_to_the_gpu_processor(monkeypatch):
    config = OptionConfig(device_preprocess=True, fast_jpeg_decode=True)
    model, calls = encoding_calls(monkeypatch, config, "cuda")
    assert model.device_preprocess
    assert calls == {"device_images": ("cuda", 50176), "processor_device": "cuda"}


@pytest.mark.parametrize(
    "config,device",
    [
        (OptionConfig(), "cuda"),
        (OptionConfig(), "cpu"),
        (OptionConfig(device_preprocess=True), "cpu"),
    ],
)
def test_stock_preprocessing_without_the_flag_or_without_cuda(monkeypatch, config, device):
    model, calls = encoding_calls(monkeypatch, config, device)
    assert not model.device_preprocess
    assert calls == {"open_images": None}


def test_defaults_load_exactly_as_before(tmp_path, monkeypatch):
    model = OptionModel.load(mode_checkpoint(tmp_path, monkeypatch), device="cpu")
    assert model.encoder.config == OptionConfig()
    assert model.vision_graphs is None and not model.budget_override
    assert not model.device_preprocess


def test_load_flags_fall_back_to_the_stock_path_on_cpu(tmp_path, monkeypatch):
    folder = mode_checkpoint(tmp_path, monkeypatch)
    model = OptionModel.load(folder, device="cpu", device_preprocess=True, vision_graph=True)
    assert model.encoder.config.device_preprocess is True
    assert not model.device_preprocess and model.vision_graphs is None
    for bad in (-1, 1.5, "yes"):
        with pytest.raises(ValueError, match="vision_graph"):
            OptionModel.load(folder, device="cpu", vision_graph=bad)
    with pytest.raises(ValueError, match="boolean"):
        OptionModel.load(folder, device="cpu", device_preprocess=1)


def test_pixel_budget_overrides_apply_after_the_mode(tmp_path, monkeypatch):
    folder = mode_checkpoint(
        tmp_path,
        monkeypatch,
        modes={"accurate": {"image_max_pixels": 1638400}},
        default_mode="accurate",
    )
    guard = OptionModel.load(folder, device="cpu", image_max_pixels=50176, image_min_pixels=1024)
    assert guard.serving_mode == "accurate" and guard.budget_override
    size = guard.encoder.processor.image_processor.size
    assert size == {"shortest_edge": 1024, "longest_edge": 50176}
    with pytest.raises(ValueError, match="inference only"):
        guard.save(tmp_path / "copy", {"completed": True, "optimizer_steps": 1})
    # Without a smaller minimum, the manifest's 65,536 would exceed a 12,544 maximum.
    with pytest.raises(ValueError, match="pixel budget"):
        OptionModel.load(folder, device="cpu", image_max_pixels=12544)
    with pytest.raises(ValueError, match="pixel budget"):
        OptionModel.load(folder, device="cpu", image_min_pixels=512)
    with pytest.raises(ValueError, match="integer"):
        OptionModel.load(folder, device="cpu", image_max_pixels=50176.0)


def test_modes_may_set_a_smaller_image_minimum(tmp_path, monkeypatch):
    modes = {
        "accurate": {"image_max_pixels": 1638400},
        "guard": {"image_max_pixels": 50176, "image_min_pixels": 1024},
    }
    folder = mode_checkpoint(tmp_path, monkeypatch, modes=modes, default_mode="accurate")
    guard = OptionModel.load(folder, device="cpu", mode="guard")
    size = guard.encoder.processor.image_processor.size
    assert size == {"shortest_edge": 1024, "longest_edge": 50176}
    assert not guard.budget_override
    for index, settings in enumerate(
        (
            {"image_max_pixels": 50176},
            {"image_max_pixels": 50176, "image_min_pixels": 60000},
            {"image_min_pixels": 512},
        )
    ):
        bad = mode_checkpoint(tmp_path / f"bad{index}", monkeypatch, modes={"m": settings})
        with pytest.raises(ValueError, match="outside the allowed budget"):
            OptionModel.load(bad, device="cpu")


def test_vision_graphs_run_eagerly_on_cpu_and_need_cuda_to_install():
    seen = []

    def forward(hidden_states, grid_thw=None, **kwargs):
        seen.append(kwargs)
        return hidden_states * 2

    visual = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"), forward=forward)
    graphs = VisionGraphs(visual)
    inputs = torch.ones(4, 3)
    with torch.no_grad():
        assert torch.equal(
            graphs(inputs, grid_thw=torch.tensor([[1, 2, 2]]), return_dict=True), inputs * 2
        )
    assert graphs.captures == 0 and seen == [{"return_dict": True}]
    with pytest.raises(ValueError, match="capacity"):
        VisionGraphs(visual, 0)
    with pytest.raises(ValueError, match="sdpa"):
        VisionGraphs(
            SimpleNamespace(
                config=SimpleNamespace(_attn_implementation="flash_attention_2"), forward=forward
            )
        )
    with pytest.raises(ValueError, match="CUDA"):
        enable_vision_graphs(torch.nn.Linear(2, 2))


def test_cpu_models_keep_eager_vision():
    model = OptionModel(fake_exit_backbone(OptionConfig()))
    assert model.vision_graphs is None and not model.device_preprocess


# GPU


def tiny_vision_tower():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=64,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3_5VisionModel(config).to("cuda", torch.bfloat16).eval()


@needs_cuda
def test_vision_graph_replays_bitwise_equal_eager_and_evicts_least_recent():
    visual = tiny_vision_tower()
    grids = [torch.tensor([[1, 4, 6]]), torch.tensor([[1, 8, 2]]), torch.tensor([[1, 4, 4]] * 2)]
    inputs = [
        torch.randn(int(g.prod(-1).sum()), 3 * 2 * 16 * 16, device="cuda", dtype=torch.bfloat16)
        for g in grids
    ]
    with torch.inference_mode():
        eager = [
            visual(x, grid_thw=g.cuda(), return_dict=True).pooler_output
            for x, g in zip(inputs, grids)
        ]
        graphs = enable_vision_graphs(visual, capacity=2)
        for _ in range(2):
            for x, g, expected in zip(inputs, grids, eager):
                out = visual(x, grid_thw=g.cuda(), return_dict=True).pooler_output
                assert torch.equal(out, expected)
                # Fresh pixels through the same graph still match eager.
                y = x.flip(0)
                assert torch.equal(
                    visual(y, grid_thw=g.cuda(), return_dict=True).pooler_output,
                    graphs.eager(y, grid_thw=g.cuda(), return_dict=True).pooler_output,
                )
    # Capacity 2 over three grids in turn: every first use of a grid recaptures.
    assert len(graphs.graphs) == 2 and graphs.captures == 6
    with torch.enable_grad():
        out = visual(inputs[0], grid_thw=grids[0].cuda(), return_dict=True).pooler_output
    assert out.requires_grad and graphs.captures == 6  # gradients bypass the graph
    # Graphs captured during inference must not block later captures outside it.
    static = torch.zeros(4, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        static.mul_(2)


@needs_cuda
@pytest.mark.parametrize("minimum,maximum", BUDGETS)
def test_gpu_resize_produces_the_stock_grid_for_every_budget(minimum, maximum):
    ip = processor()
    ip.size = {"shortest_edge": minimum, "longest_edge": maximum}
    for height, width in SIZES:
        array = pattern(height, width)
        stock = ip(images=[Image.fromarray(array)], return_tensors="pt")
        tensor = torch.from_numpy(array).cuda().permute(2, 0, 1)
        fast = ip(images=[tensor], return_tensors="pt", device="cuda")
        assert torch.equal(fast["image_grid_thw"].cpu(), stock["image_grid_thw"])
        assert fast["pixel_values"].is_cuda
        # Bicubic resampling differs slightly between CPU and GPU kernels.
        assert (fast["pixel_values"].cpu() - stock["pixel_values"]).abs().max() < 0.25


# Real checkpoints (set QEV_TEST_CHECKPOINT, optionally QEV_TEST_CACHE_DIR and QEV_TEST_DEV)

CHECKPOINT = os.environ.get("QEV_TEST_CHECKPOINT")
needs_checkpoint = pytest.mark.skipif(
    not (CUDA and CHECKPOINT), reason="needs CUDA and QEV_TEST_CHECKPOINT"
)


def load_checkpoint(**flags):
    return OptionModel.load(
        CHECKPOINT,
        device="cuda",
        cache_dir=os.environ.get("QEV_TEST_CACHE_DIR", ".cache/huggingface"),
        local_files_only=True,
        **flags,
    )


def guard_request(array, text):
    return DecisionRequest.model_validate(
        {
            "state": {"text": text, "images": [{"pixels": array}]},
            "questions": {
                "risk": {
                    "type": "choice",
                    "instructions": "Decide whether this tool call may run without asking.",
                    "criteria": {"allow": "Safe.", "block": "Dangerous.", "ask": "Unclear."},
                }
            },
        }
    )


def probabilities(result):
    return {
        name: torch.tensor(list(answer["probabilities"].values()))
        for name, answer in result["answers"].items()
    }


@needs_checkpoint
def test_vision_graph_is_exact_on_stock_pixels():
    model = load_checkpoint()
    requests = [
        guard_request(pattern(720, 1280, seed), f"Command {seed}: rm -rf ./build")
        for seed in range(3)
    ]
    stock = [model.predict(r) for r in requests]
    model.vision_graphs = enable_vision_graphs(model.encoder.model.visual)
    for _ in range(2):
        for r, ref in zip(requests, stock):
            fast = model.predict(r)
            for name, value in probabilities(fast).items():
                assert (value - probabilities(ref)[name]).abs().max().item() == 0.0
    assert model.vision_graphs.captures == 1


def dev_requests(limit):
    rows = [json.loads(line) for line in Path(os.environ["QEV_TEST_DEV"]).read_text().splitlines()]
    rows.sort(key=lambda r: hashlib.sha256(("agi2:" + r["id"]).encode()).hexdigest())
    for row in rows[:limit]:
        yield DecisionRequest.model_validate(
            {
                "state": {"text": row["state"], "images": [{"path": row["image"]}]},
                "questions": {
                    "answer": {
                        "type": "choice",
                        "instructions": row["question"],
                        "criteria": {chr(65 + i): t for i, t in enumerate(row["options"])},
                    }
                },
            }
        )


@needs_checkpoint
@pytest.mark.skipif(not os.environ.get("QEV_TEST_DEV"), reason="needs QEV_TEST_DEV items")
def test_fast_path_agrees_with_stock_on_dev_items():
    requests = list(dev_requests(5))
    model = load_checkpoint()
    stock = [model.predict(r, image_root=Path("/")) for r in requests]
    del model
    torch.cuda.empty_cache()
    fast_model = load_checkpoint(device_preprocess=True, vision_graph=True)
    assert fast_model.device_preprocess and fast_model.vision_graphs is not None
    fast = [fast_model.predict(r, image_root=Path("/")) for r in requests]
    same = sum(
        a["answers"]["answer"]["choice"] == b["answers"]["answer"]["choice"]
        for a, b in zip(stock, fast)
    )
    assert same >= 4
    for a, b in zip(stock, fast):
        assert a["usage"] == b["usage"]
        for name, value in probabilities(b).items():
            assert (value - probabilities(a)[name]).abs().max().item() < 0.1
