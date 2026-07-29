import pytest
import torch

from ecg_clinical.models import S4DClassifier, S4DLayer, XResNet1d101, trainable_parameter_count


@pytest.mark.parametrize(
    ("model", "minimum_parameters", "maximum_parameters"),
    [
        (XResNet1d101(), 1_000_000, 3_000_000),
        (S4DClassifier(d_model=64), 20_000, 500_000),
    ],
)
def test_model_forward_backward_and_parameter_scale(
    model: torch.nn.Module, minimum_parameters: int, maximum_parameters: int
) -> None:
    torch.manual_seed(7)
    model.train()
    inputs = torch.randn(2, 12, 250)
    outputs = model(inputs)
    assert outputs.shape == (2, 16)
    assert torch.isfinite(outputs).all()
    outputs.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert minimum_parameters < trainable_parameter_count(model) < maximum_parameters


def test_s4d_eval_is_deterministic() -> None:
    torch.manual_seed(11)
    model = S4DClassifier(d_model=32, d_state=4, blocks=2, dropout=0.2).eval()
    inputs = torch.randn(2, 12, 250)
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    torch.testing.assert_close(first, second)


def test_full_model_parameter_counts_are_frozen() -> None:
    assert trainable_parameter_count(XResNet1d101()) == 1_873_680
    assert trainable_parameter_count(S4DClassifier()) == 2_165_264


def test_fused_bidirectional_convolution_matches_two_causal_passes() -> None:
    torch.manual_seed(13)
    inputs = torch.randn(2, 3, 17)
    forward_kernel = torch.randn(3, 17)
    backward_kernel = torch.randn(3, 17)
    expected = S4DLayer._causal_convolution(inputs, forward_kernel)
    expected += S4DLayer._causal_convolution(inputs.flip(-1), backward_kernel).flip(-1)
    observed = S4DLayer._bidirectional_convolution(inputs, forward_kernel, backward_kernel)
    torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-5)
