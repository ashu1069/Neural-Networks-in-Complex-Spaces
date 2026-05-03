from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from cvnn.activations import (
    ComplexCardioid,
    ComplexTanh,
    CReLU,
    ModReLU,
    Siglog,
    ZReLU,
    complex_cardioid,
    complex_tanh,
    crelu,
    modrelu,
    siglog,
    zrelu,
)


def test_crelu_values() -> None:
    z = torch.tensor([-1 - 2j, 1 - 2j, -1 + 2j, 1 + 2j])

    outputs = crelu(z)

    expected = torch.tensor([0 + 0j, 1 + 0j, 0 + 2j, 1 + 2j])
    assert torch.allclose(outputs, expected)


def test_zrelu_values() -> None:
    z = torch.tensor([-1 - 2j, 1 - 2j, -1 + 2j, 1 + 2j, 0 + 0j])

    outputs = zrelu(z)

    expected = torch.tensor([0 + 0j, 0 + 0j, 0 + 0j, 1 + 2j, 0 + 0j])
    assert torch.allclose(outputs, expected)


def test_modrelu_values_and_zero_edge_case() -> None:
    z = torch.tensor([0 + 0j, 1 + 0j, 3 + 4j])

    outputs = modrelu(z, bias=-2.0)

    expected = torch.tensor([0 + 0j, 0 + 0j, 1.8 + 2.4j])
    assert torch.allclose(outputs, expected, atol=1e-6)
    assert torch.isfinite(outputs).all()


def test_complex_cardioid_values() -> None:
    z = torch.tensor([1 + 0j, -1 + 0j, 0 + 1j, 0 + 0j])

    outputs = complex_cardioid(z)

    expected = torch.tensor([1 + 0j, 0 + 0j, 0 + 0.5j, 0 + 0j])
    assert torch.allclose(outputs, expected, atol=1e-6)


def test_siglog_is_bounded_by_one() -> None:
    z = torch.tensor([0 + 0j, 1 + 0j, 3 + 4j, 10 - 10j])

    outputs = siglog(z)

    assert torch.all(outputs.abs() < 1.0)
    assert outputs[0] == 0


def test_complex_tanh_matches_torch_tanh() -> None:
    z = torch.tensor([0.2 + 0.3j, -0.4 + 0.1j])

    assert torch.allclose(complex_tanh(z), torch.tanh(z))


@pytest.mark.parametrize(
    "module",
    [CReLU(), ZReLU(), ModReLU(), ComplexCardioid(), Siglog(), ComplexTanh()],
)
def test_activation_modules_preserve_shape_and_complex_dtype(
    module: torch.nn.Module,
) -> None:
    z = torch.randn(4, 3, dtype=torch.complex64)

    outputs = module(z)

    assert outputs.shape == z.shape
    assert torch.is_complex(outputs)


def test_modrelu_module_bias_gets_gradient() -> None:
    module = ModReLU(init_bias=0.1)
    z = torch.randn(8, dtype=torch.complex64)

    loss = module(z).abs().square().mean()
    loss.backward()

    assert module.bias.grad is not None
    assert torch.isfinite(module.bias.grad).all()


@pytest.mark.parametrize(
    "activation",
    [
        lambda z: modrelu(z, bias=0.5),
        complex_cardioid,
        siglog,
        complex_tanh,
    ],
)
def test_smooth_activation_gradcheck(activation: object) -> None:
    z = (torch.randn(5, dtype=torch.complex128) + 1.5 + 1.5j).requires_grad_(True)

    def loss_fn(input_: torch.Tensor) -> torch.Tensor:
        outputs = activation(input_)  # type: ignore[operator]
        return outputs.abs().square().sum()

    assert gradcheck(loss_fn, (z,), eps=1e-6, atol=1e-4)
