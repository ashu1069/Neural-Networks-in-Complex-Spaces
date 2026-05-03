"""Complex-valued 1D convolution."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from cvnn.init import ComplexDistribution, complex_xavier_uniform_, complex_zeros_


class ComplexConv1d(nn.Module):
    """A 1D convolution for complex-valued tensors.

    PyTorch's `F.conv1d` natively supports `torch.complex64` / `torch.complex128`
    on CPU, MPS, and CUDA (verified by `scripts/check_torch_complex_support.py`),
    so this layer is a thin wrapper over the functional form rather than a
    matmul-based fallback like `ComplexLinear` needed.

    Inputs:  `(B, in_channels, L)` complex.
    Outputs: `(B, out_channels, L_out)` complex.
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    groups: int
    weight: Parameter
    bias: Parameter | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        init_distribution: ComplexDistribution = "rectangular",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            msg = "in_channels and out_channels must be positive"
            raise ValueError(msg)
        if kernel_size <= 0:
            msg = "kernel_size must be positive"
            raise ValueError(msg)
        if in_channels % groups != 0 or out_channels % groups != 0:
            msg = "in_channels and out_channels must both be divisible by groups"
            raise ValueError(msg)

        dtype = _resolve_complex_dtype(dtype)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.init_distribution = init_distribution
        self.weight = Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                kernel_size,
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            self.bias = Parameter(torch.empty(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        complex_xavier_uniform_(self.weight, distribution=self.init_distribution)
        if self.bias is not None:
            complex_zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv1d(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups}, "
            f"bias={self.bias is not None}"
        )


def _resolve_complex_dtype(dtype: torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return torch.complex64
    if not dtype.is_complex:
        msg = f"ComplexConv1d requires a complex dtype, got {dtype}"
        raise TypeError(msg)
    return dtype


__all__ = ["ComplexConv1d"]
