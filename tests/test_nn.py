from __future__ import annotations

import pytest
import torch
from torch import nn as torch_nn

from cvnn.layers import ComplexDropout, ComplexLinear
from cvnn.nn import ComplexMLP


def test_complex_mlp_forward_shape_and_dtype() -> None:
    model = ComplexMLP(3, [5, 4], 2, dtype=torch.complex64)
    inputs = torch.randn(7, 3, dtype=torch.complex64)

    outputs = model(inputs)

    assert outputs.shape == (7, 2)
    assert outputs.dtype == torch.complex64


class _Identity(torch_nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def test_complex_mlp_inserts_activation_between_hidden_layers_only() -> None:
    activation_calls = 0

    def factory() -> torch_nn.Module:
        nonlocal activation_calls
        activation_calls += 1
        return _Identity()

    model = ComplexMLP(3, [5, 4], 2, activation_factory=factory, dtype=torch.complex64)

    linear_layers = [m for m in model.net if isinstance(m, ComplexLinear)]
    activations = [m for m in model.net if isinstance(m, _Identity)]
    assert len(linear_layers) == 3
    assert len(activations) == 2
    assert activation_calls == 2
    assert isinstance(model.net[-1], ComplexLinear)


def test_complex_mlp_inserts_dropout_between_hidden_layers_only() -> None:
    model = ComplexMLP(3, [5, 4], 2, dropout=0.5, dtype=torch.complex64)

    dropouts = [m for m in model.net if isinstance(m, ComplexDropout)]
    assert len(dropouts) == 2
    assert isinstance(model.net[-1], ComplexLinear)


def test_complex_mlp_dropout_off_in_eval() -> None:
    torch.manual_seed(0)
    model = ComplexMLP(4, [16], 4, dropout=0.5, dtype=torch.complex64).eval()
    inputs = torch.randn(8, 4, dtype=torch.complex64)

    out_a = model(inputs)
    out_b = model(inputs)

    assert torch.equal(out_a, out_b)


def test_complex_mlp_propagates_bias_false() -> None:
    model = ComplexMLP(3, [5], 2, bias=False, dtype=torch.complex64)

    for layer in model.net:
        if isinstance(layer, ComplexLinear):
            assert layer.bias is None


def test_complex_mlp_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        ComplexMLP(3, [0, 4], 2, dtype=torch.complex64)
    with pytest.raises(ValueError):
        ComplexMLP(-1, [4], 2, dtype=torch.complex64)


def test_complex_mlp_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError):
        ComplexMLP(3, [4], 2, dropout=-0.1, dtype=torch.complex64)
    with pytest.raises(ValueError):
        ComplexMLP(3, [4], 2, dropout=1.5, dtype=torch.complex64)
