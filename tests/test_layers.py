from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.autograd import gradcheck

from cvnn.layers import ComplexDropout, ComplexLinear, complex_linear


def test_complex_linear_shape_and_dtype() -> None:
    layer = ComplexLinear(4, 3, dtype=torch.complex64)
    inputs = torch.randn(5, 4, dtype=torch.complex64)

    outputs = layer(inputs)

    assert outputs.shape == (5, 3)
    assert outputs.dtype == torch.complex64


def test_complex_linear_supports_batched_inputs() -> None:
    layer = ComplexLinear(4, 3, dtype=torch.complex64)
    inputs = torch.randn(2, 5, 4, dtype=torch.complex64)

    outputs = layer(inputs)

    assert outputs.shape == (2, 5, 3)


def test_complex_linear_gradcheck_cpu() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(2, 3, dtype=torch.complex128, requires_grad=True)
    weight = torch.randn(4, 3, dtype=torch.complex128, requires_grad=True)
    bias = torch.randn(4, dtype=torch.complex128, requires_grad=True)

    def loss_fn(
        input_: torch.Tensor,
        weight_: torch.Tensor,
        bias_: torch.Tensor,
    ) -> torch.Tensor:
        outputs = complex_linear(input_, weight_, bias_)
        return outputs.real.square().sum() + outputs.imag.square().sum()

    assert gradcheck(loss_fn, (inputs, weight, bias), eps=1e-6, atol=1e-4)


def test_complex_linear_real_only_equivalence() -> None:
    torch.manual_seed(0)
    real_layer = nn.Linear(4, 3)
    complex_layer = ComplexLinear(4, 3, dtype=torch.complex64)
    with torch.no_grad():
        complex_weight = torch.complex(
            real_layer.weight,
            torch.zeros_like(real_layer.weight),
        )
        complex_bias = torch.complex(real_layer.bias, torch.zeros_like(real_layer.bias))
        complex_layer.weight.copy_(complex_weight)
        complex_layer.bias.copy_(complex_bias)

    real_inputs = torch.randn(5, 4)
    complex_inputs = torch.complex(real_inputs, torch.zeros_like(real_inputs))

    real_outputs = real_layer(real_inputs)
    complex_outputs = complex_layer(complex_inputs)

    assert torch.allclose(complex_outputs.real, real_outputs, atol=1e-6)
    assert torch.allclose(
        complex_outputs.imag,
        torch.zeros_like(real_outputs),
        atol=1e-6,
    )


@pytest.mark.parametrize("device_name", ["mps", "cuda"])
def test_complex_linear_accelerator_agrees_with_cpu(device_name: str) -> None:
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    device = torch.device(device_name)
    cpu_layer = ComplexLinear(4, 3, dtype=torch.complex64).eval()
    device_layer = ComplexLinear(4, 3, dtype=torch.complex64, device=device).eval()
    device_layer.load_state_dict(cpu_layer.state_dict())
    inputs = torch.randn(5, 4, dtype=torch.complex64)

    cpu_outputs = cpu_layer(inputs)
    device_outputs = device_layer(inputs.to(device)).cpu()

    assert torch.allclose(device_outputs, cpu_outputs, atol=1e-5, rtol=1e-4)


def test_complex_linear_without_bias() -> None:
    layer = ComplexLinear(4, 3, bias=False, dtype=torch.complex64)
    inputs = torch.randn(5, 4, dtype=torch.complex64)

    outputs = layer(inputs)

    assert layer.bias is None
    assert outputs.shape == (5, 3)
    expected = inputs @ layer.weight.transpose(-2, -1)
    assert torch.allclose(outputs, expected)


def test_complex_linear_complex128_dtype() -> None:
    layer = ComplexLinear(4, 3, dtype=torch.complex128)
    inputs = torch.randn(5, 4, dtype=torch.complex128)

    outputs = layer(inputs)

    assert outputs.dtype == torch.complex128
    assert layer.weight.dtype == torch.complex128
    assert layer.bias is not None
    assert layer.bias.dtype == torch.complex128


def test_complex_linear_rejects_real_dtype() -> None:
    with pytest.raises(TypeError):
        ComplexLinear(4, 3, dtype=torch.float32)


def test_complex_linear_rejects_nonpositive_features() -> None:
    with pytest.raises(ValueError):
        ComplexLinear(0, 3)
    with pytest.raises(ValueError):
        ComplexLinear(4, -1)


def test_complex_dropout_eval_is_identity() -> None:
    dropout = ComplexDropout(p=0.5).eval()
    inputs = torch.randn(8, dtype=torch.complex64)

    outputs = dropout(inputs)

    assert torch.equal(outputs, inputs)


def test_complex_dropout_uses_real_mask() -> None:
    torch.manual_seed(0)
    dropout = ComplexDropout(p=0.5).train()
    inputs = torch.full((64,), 1.0 + 2.0j, dtype=torch.complex64)

    outputs = dropout(inputs)
    scale = outputs / inputs

    assert torch.allclose(scale.imag, torch.zeros_like(scale.imag))
    assert set(scale.real.unique().tolist()).issubset({0.0, 2.0})


def test_complex_dropout_p_zero_is_identity_in_train() -> None:
    dropout = ComplexDropout(p=0.0).train()
    inputs = torch.randn(8, dtype=torch.complex64)

    assert torch.equal(dropout(inputs), inputs)


def test_complex_dropout_gradcheck_p_zero() -> None:
    dropout = ComplexDropout(p=0.0).train()
    inputs = torch.randn(6, dtype=torch.complex128, requires_grad=True)

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        out = dropout(x)
        return out.real.square().sum() + out.imag.square().sum()

    assert gradcheck(loss_fn, (inputs,), eps=1e-6, atol=1e-4)


def test_complex_dropout_rejects_invalid_p() -> None:
    with pytest.raises(ValueError):
        ComplexDropout(p=-0.1)
    with pytest.raises(ValueError):
        ComplexDropout(p=1.5)
