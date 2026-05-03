from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.autograd import gradcheck

from cvnn.layers import ComplexConv1d


def test_complex_conv1d_shape_and_dtype() -> None:
    layer = ComplexConv1d(2, 3, kernel_size=5, padding=2, dtype=torch.complex64)
    inputs = torch.randn(4, 2, 16, dtype=torch.complex64)

    outputs = layer(inputs)

    assert outputs.shape == (4, 3, 16)
    assert outputs.dtype == torch.complex64


def test_complex_conv1d_no_padding_truncates_length() -> None:
    layer = ComplexConv1d(1, 2, kernel_size=3, padding=0, dtype=torch.complex64)
    inputs = torch.randn(2, 1, 10, dtype=torch.complex64)

    outputs = layer(inputs)

    assert outputs.shape == (2, 2, 8)


def test_complex_conv1d_without_bias() -> None:
    layer = ComplexConv1d(2, 3, kernel_size=3, padding=1, bias=False)
    inputs = torch.randn(2, 2, 8, dtype=torch.complex64)

    outputs = layer(inputs)

    assert layer.bias is None
    assert outputs.shape == (2, 3, 8)


def test_complex_conv1d_real_only_equivalence() -> None:
    torch.manual_seed(0)
    real_layer = nn.Conv1d(2, 3, kernel_size=3, padding=1)
    complex_layer = ComplexConv1d(2, 3, kernel_size=3, padding=1, dtype=torch.complex64)
    with torch.no_grad():
        complex_layer.weight.copy_(
            torch.complex(real_layer.weight, torch.zeros_like(real_layer.weight))
        )
        assert complex_layer.bias is not None
        complex_layer.bias.copy_(
            torch.complex(real_layer.bias, torch.zeros_like(real_layer.bias))
        )

    real_inputs = torch.randn(4, 2, 16)
    complex_inputs = torch.complex(real_inputs, torch.zeros_like(real_inputs))

    real_outputs = real_layer(real_inputs)
    complex_outputs = complex_layer(complex_inputs)

    assert torch.allclose(complex_outputs.real, real_outputs, atol=1e-6)
    assert torch.allclose(
        complex_outputs.imag, torch.zeros_like(real_outputs), atol=1e-6
    )


def test_complex_conv1d_gradcheck_cpu() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(2, 2, 6, dtype=torch.complex128, requires_grad=True)
    weight = torch.randn(3, 2, 3, dtype=torch.complex128, requires_grad=True)
    bias = torch.randn(3, dtype=torch.complex128, requires_grad=True)

    def loss_fn(
        input_: torch.Tensor,
        weight_: torch.Tensor,
        bias_: torch.Tensor,
    ) -> torch.Tensor:
        outputs = torch.nn.functional.conv1d(input_, weight_, bias_, padding=1)
        return outputs.real.square().sum() + outputs.imag.square().sum()

    assert gradcheck(loss_fn, (inputs, weight, bias), eps=1e-6, atol=1e-4)


def test_complex_conv1d_rejects_real_dtype() -> None:
    with pytest.raises(TypeError):
        ComplexConv1d(2, 3, kernel_size=3, dtype=torch.float32)


def test_complex_conv1d_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        ComplexConv1d(0, 3, kernel_size=3)
    with pytest.raises(ValueError):
        ComplexConv1d(2, 0, kernel_size=3)
    with pytest.raises(ValueError):
        ComplexConv1d(2, 3, kernel_size=0)
    with pytest.raises(ValueError):
        ComplexConv1d(3, 4, kernel_size=3, groups=2)


@pytest.mark.parametrize("device_name", ["mps", "cuda"])
def test_complex_conv1d_accelerator_agrees_with_cpu(device_name: str) -> None:
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    device = torch.device(device_name)
    cpu_layer = ComplexConv1d(2, 3, kernel_size=5, padding=2).eval()
    device_layer = ComplexConv1d(2, 3, kernel_size=5, padding=2, device=device).eval()
    device_layer.load_state_dict(cpu_layer.state_dict())
    inputs = torch.randn(4, 2, 16, dtype=torch.complex64)

    cpu_outputs = cpu_layer(inputs)
    device_outputs = device_layer(inputs.to(device)).cpu()

    assert torch.allclose(device_outputs, cpu_outputs, atol=1e-5, rtol=1e-4)
