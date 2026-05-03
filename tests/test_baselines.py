from __future__ import annotations

import pytest
import torch

from cvnn.baselines import (
    RealStackedLinear,
    complex_to_real_reparam,
    count_real_parameters,
)
from cvnn.layers import ComplexLinear


def test_real_stacked_linear_shape() -> None:
    layer = RealStackedLinear(4, 3)
    inputs = torch.randn(5, 8)

    outputs = layer(inputs)

    assert outputs.shape == (5, 6)


def test_real_stacked_linear_rejects_nonpositive_features() -> None:
    with pytest.raises(ValueError):
        RealStackedLinear(0, 3)
    with pytest.raises(ValueError):
        RealStackedLinear(4, -1)


def test_count_real_parameters_doubles_complex_numel() -> None:
    complex_layer = ComplexLinear(4, 3)
    real_layer = RealStackedLinear(4, 3)

    assert count_real_parameters(complex_layer) == 2 * (4 * 3 + 3)
    assert count_real_parameters(real_layer) == (2 * 4) * (2 * 3) + (2 * 3)


def test_complex_to_real_reparam_matches_complex_forward() -> None:
    torch.manual_seed(0)
    complex_layer = ComplexLinear(4, 3, dtype=torch.complex64)
    real_layer = complex_to_real_reparam(complex_layer)

    z = torch.randn(7, 4, dtype=torch.complex64)
    complex_out = complex_layer(z)
    real_in = torch.cat([z.real, z.imag], dim=-1)
    real_out = real_layer(real_in)

    expected_real = complex_out.real
    expected_imag = complex_out.imag
    actual_real = real_out[..., :3]
    actual_imag = real_out[..., 3:]
    assert torch.allclose(actual_real, expected_real, atol=1e-6)
    assert torch.allclose(actual_imag, expected_imag, atol=1e-6)


def test_complex_to_real_reparam_without_bias() -> None:
    complex_layer = ComplexLinear(3, 2, bias=False, dtype=torch.complex64)
    real_layer = complex_to_real_reparam(complex_layer)

    z = torch.randn(5, 3, dtype=torch.complex64)
    real_in = torch.cat([z.real, z.imag], dim=-1)
    complex_out = complex_layer(z)
    real_out = real_layer(real_in)

    assert real_layer.bias is None
    assert torch.allclose(real_out[..., :2], complex_out.real, atol=1e-6)
    assert torch.allclose(real_out[..., 2:], complex_out.imag, atol=1e-6)
