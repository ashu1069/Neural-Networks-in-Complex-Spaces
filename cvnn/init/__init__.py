"""Complex-aware weight initializers."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

FanMode = Literal["fan_in", "fan_out"]
ComplexDistribution = Literal["rectangular", "polar"]


def complex_xavier_uniform_(
    tensor: Tensor,
    *,
    gain: float = 1.0,
    distribution: ComplexDistribution = "rectangular",
) -> Tensor:
    """Fill a complex tensor with a Glorot-style initialization."""

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    target_second_moment = gain**2 * 2.0 / (fan_in + fan_out)
    return _complex_uniform_(tensor, target_second_moment, distribution)


def complex_kaiming_uniform_(
    tensor: Tensor,
    *,
    mode: FanMode = "fan_in",
    gain: float = math.sqrt(2.0),
    distribution: ComplexDistribution = "rectangular",
) -> Tensor:
    """Fill a complex tensor with a He/Kaiming-style initialization."""

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    target_second_moment = gain**2 / fan
    return _complex_uniform_(tensor, target_second_moment, distribution)


def complex_zeros_(tensor: Tensor) -> Tensor:
    """Fill a real or complex tensor with zeros."""

    with torch.no_grad():
        return tensor.zero_()


def _complex_uniform_(
    tensor: Tensor,
    target_second_moment: float,
    distribution: ComplexDistribution,
) -> Tensor:
    if not torch.is_complex(tensor):
        msg = "complex initializers expect a complex tensor"
        raise TypeError(msg)
    if distribution == "rectangular":
        return _rectangular_uniform_(tensor, target_second_moment)
    if distribution == "polar":
        return _polar_rayleigh_uniform_phase_(tensor, target_second_moment)
    msg = f"unsupported complex initializer distribution: {distribution}"
    raise ValueError(msg)


def _rectangular_uniform_(tensor: Tensor, target_second_moment: float) -> Tensor:
    bound = math.sqrt(1.5 * target_second_moment)
    with torch.no_grad():
        tensor.real.uniform_(-bound, bound)
        tensor.imag.uniform_(-bound, bound)
    return tensor


def _polar_rayleigh_uniform_phase_(
    tensor: Tensor,
    target_second_moment: float,
) -> Tensor:
    sigma = math.sqrt(target_second_moment / 2.0)
    real_dtype = tensor.real.dtype
    with torch.no_grad():
        uniform = torch.rand(tensor.shape, dtype=real_dtype, device=tensor.device)
        radius = sigma * torch.sqrt(-2.0 * torch.log1p(-uniform.clamp_max(1 - 1e-12)))
        phase = torch.empty(tensor.shape, dtype=real_dtype, device=tensor.device)
        phase.uniform_(0.0, 2.0 * math.pi)
        tensor.copy_(torch.polar(radius, phase))
    return tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    dimensions = tensor.dim()
    if dimensions < 2:
        msg = "fan in and fan out require a tensor with at least 2 dimensions"
        raise ValueError(msg)

    num_output_fmaps = tensor.size(0)
    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = math.prod(tensor.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


__all__ = [
    "ComplexDistribution",
    "FanMode",
    "complex_kaiming_uniform_",
    "complex_xavier_uniform_",
    "complex_zeros_",
]
