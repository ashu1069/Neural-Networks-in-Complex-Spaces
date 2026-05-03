from __future__ import annotations

import pytest
import torch

from cvnn.init import complex_kaiming_uniform_, complex_xavier_uniform_


@pytest.mark.parametrize("distribution", ["rectangular", "polar"])
def test_complex_xavier_uniform_fills_complex_tensor(distribution: str) -> None:
    tensor = torch.empty(64, 32, dtype=torch.complex64)

    complex_xavier_uniform_(tensor, distribution=distribution)

    assert torch.is_complex(tensor)
    assert torch.isfinite(tensor.real).all()
    assert torch.isfinite(tensor.imag).all()
    assert tensor.abs().sum() > 0


@pytest.mark.parametrize("distribution", ["rectangular", "polar"])
def test_complex_kaiming_uniform_fills_complex_tensor(distribution: str) -> None:
    tensor = torch.empty(64, 32, dtype=torch.complex64)

    complex_kaiming_uniform_(tensor, distribution=distribution)

    assert torch.is_complex(tensor)
    assert torch.isfinite(tensor.real).all()
    assert torch.isfinite(tensor.imag).all()
    assert tensor.abs().sum() > 0


def test_complex_initializers_reject_real_tensor() -> None:
    tensor = torch.empty(8, 4)

    with pytest.raises(TypeError):
        complex_xavier_uniform_(tensor)


@pytest.mark.parametrize("distribution", ["rectangular", "polar"])
def test_complex_xavier_uniform_targets_second_moment(distribution: str) -> None:
    torch.manual_seed(0)
    fan_out, fan_in = 128, 64
    tensor = torch.empty(fan_out, fan_in, dtype=torch.complex128)

    complex_xavier_uniform_(tensor, distribution=distribution)

    target = 2.0 / (fan_in + fan_out)
    measured = tensor.abs().square().mean().item()
    assert abs(measured - target) / target < 0.1


@pytest.mark.parametrize("distribution", ["rectangular", "polar"])
@pytest.mark.parametrize("mode", ["fan_in", "fan_out"])
def test_complex_kaiming_uniform_targets_second_moment(
    distribution: str,
    mode: str,
) -> None:
    import math

    torch.manual_seed(0)
    fan_out, fan_in = 128, 64
    tensor = torch.empty(fan_out, fan_in, dtype=torch.complex128)
    gain = math.sqrt(2.0)

    complex_kaiming_uniform_(tensor, mode=mode, gain=gain, distribution=distribution)  # type: ignore[arg-type]

    fan = fan_in if mode == "fan_in" else fan_out
    target = gain**2 / fan
    measured = tensor.abs().square().mean().item()
    assert abs(measured - target) / target < 0.1
