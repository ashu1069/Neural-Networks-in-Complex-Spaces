"""Module wrappers for complex-valued activations."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from cvnn.activations.functions import (
    complex_cardioid,
    complex_tanh,
    crelu,
    modrelu,
    siglog,
    zrelu,
)


class CReLU(nn.Module):
    """Split ReLU applied independently to real and imaginary parts."""

    def forward(self, input: Tensor) -> Tensor:
        return crelu(input)


class ZReLU(nn.Module):
    """Phase-gated ReLU that keeps values in the closed first quadrant."""

    def forward(self, input: Tensor) -> Tensor:
        return zrelu(input)


class ModReLU(nn.Module):
    """Magnitude-gated ReLU with a learnable real bias.

    The bias broadcasts against the **last** dimension of the input. That is
    correct for `(B, F)` MLPs and `(B, T, F)` transformer-style sequences but
    *not* for `(B, C, H, W)` conv layouts where the channel axis is dim 1 -
    permute or reshape before applying. See `modrelu` for the bias-sign
    regime: `init_bias <= 0` keeps the activation continuous at the origin.
    """

    bias: Parameter

    def __init__(
        self,
        num_features: int | None = None,
        *,
        init_bias: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_features is not None and num_features <= 0:
            msg = "num_features must be positive when provided"
            raise ValueError(msg)
        shape = () if num_features is None else (num_features,)
        real_dtype = _resolve_real_dtype(dtype)
        self.bias = Parameter(
            torch.full(shape, init_bias, device=device, dtype=real_dtype)
        )

    def forward(self, input: Tensor) -> Tensor:
        return modrelu(input, self.bias)

    def extra_repr(self) -> str:
        return f"num_features={self.bias.numel() if self.bias.dim() else None}"


class ComplexCardioid(nn.Module):
    """Cardioid activation."""

    def forward(self, input: Tensor) -> Tensor:
        return complex_cardioid(input)


class Siglog(nn.Module):
    """Bounded smooth `z / (1 + |z|)` activation."""

    def forward(self, input: Tensor) -> Tensor:
        return siglog(input)


class ComplexTanh(nn.Module):
    """Complex tanh cautionary baseline."""

    def forward(self, input: Tensor) -> Tensor:
        return complex_tanh(input)


def _resolve_real_dtype(dtype: torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    if dtype.is_floating_point:
        return dtype
    msg = f"ModReLU bias requires a floating or complex dtype, got {dtype}"
    raise TypeError(msg)


__all__ = [
    "CReLU",
    "ComplexCardioid",
    "ComplexTanh",
    "ModReLU",
    "Siglog",
    "ZReLU",
]
