import pytest
import torch

from veyra.cuda_graph import ShortInputGraphs, enable_short_input_graphs, pad_to_bucket


def tiny_text_model(layers=4):
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    torch.manual_seed(0)
    return Qwen3_5TextModel(config).eval()


@pytest.mark.parametrize("rows", [None, 3, 4])
def test_right_padding_leaves_real_positions_unchanged(rows):
    model = tiny_text_model()
    embeds = torch.randn(1, 70, 32)  # crosses a 64-token gated-delta chunk
    positions = None
    if rows:
        positions = torch.arange(70).view(1, 1, -1).expand(rows, 1, -1).contiguous()
        positions[-1] = positions[-1] // 2  # an image-like rotary section
    with torch.no_grad():
        plain = model(inputs_embeds=embeds, position_ids=positions, use_cache=False)
        padded, padded_positions = pad_to_bucket(embeds, positions, 128)
        tail = model(inputs_embeds=padded, position_ids=padded_positions, use_cache=False)
    assert padded.shape == (1, 128, 32) and padded_positions.shape[-1] == 128
    assert torch.allclose(plain.last_hidden_state, tail.last_hidden_state[:, :70], atol=1e-5)


def test_graphs_fall_back_to_eager_off_cuda_and_for_long_inputs():
    model = tiny_text_model()
    eager = model.forward
    graphs = ShortInputGraphs(model, buckets=(64,))
    embeds = torch.randn(1, 10, 32)
    with torch.no_grad():
        assert torch.equal(
            graphs(inputs_embeds=embeds, use_cache=False).last_hidden_state,
            eager(inputs_embeds=embeds, use_cache=False).last_hidden_state,
        )
        graphs(inputs_embeds=torch.randn(1, 65, 32), use_cache=False)
    assert graphs.graphs == {}
    with pytest.raises(ValueError, match="CUDA device"):
        enable_short_input_graphs(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_replay_matches_padded_eager_on_cuda():
    model = tiny_text_model().cuda()
    eager = model.forward
    graphs = enable_short_input_graphs(model, buckets=(64, 128))
    assert set(graphs.graphs) == {(64, 4), (128, 4)}
    with torch.inference_mode():
        for length in (10, 70, 70):
            embeds = torch.randn(1, length, 32, device="cuda")
            out = model(inputs_embeds=embeds, use_cache=False).last_hidden_state
            padded, positions = pad_to_bucket(embeds, None, 64 if length <= 64 else 128)
            reference = eager(inputs_embeds=padded, position_ids=positions, use_cache=False)
            assert torch.allclose(out, reference.last_hidden_state[:, :length], atol=1e-5)
        long = torch.randn(1, 129, 32, device="cuda")
        assert torch.allclose(
            model(inputs_embeds=long, use_cache=False).last_hidden_state,
            eager(inputs_embeds=long, use_cache=False).last_hidden_state,
        )
    assert set(graphs.graphs) == {(64, 4), (128, 4)}
