import pytest
import torch

from veyra.binding_head import ConditionedReadout


def test_initial_residual_is_exactly_zero():
    torch.manual_seed(5)
    head = ConditionedReadout(rank=4, input_size=8).eval()
    residual = head(torch.randn(5, 8), torch.randn(5, 3))
    assert torch.equal(residual, torch.zeros(5, 16))


def test_predicted_condition_changes_the_neural_output():
    torch.manual_seed(6)
    head = ConditionedReadout(rank=4, input_size=8).eval()
    with torch.no_grad():
        head.experts.weight.normal_()
    state = torch.randn(1, 8).repeat(2, 1)
    conditions = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
    output = head(state, conditions)
    assert not torch.allclose(output[0], output[1])
    head.mode = "uniform"
    control = head(state, conditions)
    assert torch.equal(control[0], control[1])


def test_head_can_learn_from_decision_targets_without_oracle_condition_input():
    head = ConditionedReadout(rank=4, input_size=8)
    state = torch.randn(3, 8)
    conditions = torch.randn(3, 3)
    target = torch.tensor([0, 1, 2])
    loss = torch.nn.functional.cross_entropy(head(state, conditions), target)
    loss.backward()
    assert head.experts.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("rank", [0, -1, 257, True])
def test_binding_rank_is_bounded(rank):
    with pytest.raises(ValueError, match="rank"):
        ConditionedReadout(rank=rank)


def test_binding_mode_and_gate_shape_are_explicit():
    with pytest.raises(ValueError, match="mode"):
        ConditionedReadout(mode="oracle")
    with pytest.raises(ValueError, match="three predicted"):
        ConditionedReadout(input_size=8)(torch.zeros(2, 8), torch.zeros(2, 2))
