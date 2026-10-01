import pytest
import torch

from veyra.average_adapters import average_states


def state(rank):
    return {
        "readout.weight": torch.randn(3, 5),
        "layer.lora_a": torch.randn(rank, 5),
        "layer.lora_b": torch.randn(4, rank),
    }


def test_factor_concatenation_matches_weighted_updates_without_cross_terms():
    torch.manual_seed(31)
    first, second = state(2), state(3)
    averaged = average_states([first, second], [2.0, 4.0], [1.0, 3.0])
    expected = (0.25 * 2 * first["layer.lora_b"] @ first["layer.lora_a"]) + (
        0.75 * 4 * second["layer.lora_b"] @ second["layer.lora_a"]
    )
    actual = averaged["layer.lora_b"] @ averaged["layer.lora_a"]
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(
        averaged["readout.weight"], 0.25 * first["readout.weight"] + 0.75 * second["readout.weight"]
    )


@pytest.mark.parametrize("weights", [[-1, 2], [0, 1], [float("nan"), 1], [1]])
def test_average_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        average_states([state(2), state(2)], [2.0, 2.0], weights)
