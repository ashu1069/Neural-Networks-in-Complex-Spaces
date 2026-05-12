"""Rotation-equivariant real-valued Conv1d baseline.

Proposition 3 of the paper characterizes complex layers as exactly the
$U(1)$-equivariant subspace of stacked-real $2$-channel layers: a real
$2\\times 2$ kernel block commutes with $SO(2)$ iff it has the form
$aI + bJ$, which is the $\\mathbb{R}$-algebra image of multiplication by
$a + ib$. This module implements that subspace as a real-valued Conv1d
whose kernel taps are constrained to that 2-parameter form, and provides
a numerical-equivalence helper that materializes a `ComplexConv1d` as
this real layer (and vice versa). Used as the "complex inductive bias,
implemented in real coordinates" control baseline that isolates the
equivariance constraint from the complex-tensor implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from cvnn.layers.conv import ComplexConv1d


class RotationEquivariantConv1d(nn.Module):
    """Real-valued Conv1d whose 2-channel kernel taps lie in the $aI+bJ$ subspace.

    Channel layout follows `RealStackedLinear`: the input channel axis of
    length `2 * in_channels` is ordered as
    `[Re_0, ..., Re_{c-1}, Im_0, ..., Im_{c-1}]`, and likewise on the
    output. Trainable parameters are two real tensors `a, b` of shape
    `(out_channels, in_channels, kernel_size)`. The effective real
    weight tensor of shape `(2*out, 2*in, K)` is reconstructed at every
    forward pass as

        W[o,        i,        :] =  a[o,i,:]
        W[o,        i + in_c,  :] = -b[o,i,:]
        W[o + out_c, i,        :] =  b[o,i,:]
        W[o + out_c, i + in_c, :] =  a[o,i,:]

    so that the layer is, by construction and by Proposition 3,
    isomorphic to a `ComplexConv1d(in_channels, out_channels, K)` with
    complex weight `a + i b` and complex bias `b_re + i b_im`.
    Equivalence is asserted in `tests/test_rotation_equivariant.py`.
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    groups: int

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
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            msg = "in_channels and out_channels must be positive"
            raise ValueError(msg)
        if kernel_size <= 0:
            msg = "kernel_size must be positive"
            raise ValueError(msg)
        if groups != 1:
            msg = "RotationEquivariantConv1d currently supports groups=1 only"
            raise NotImplementedError(msg)

        if dtype is None:
            dtype = torch.float32
        if dtype.is_complex:
            msg = "RotationEquivariantConv1d uses real dtype; pass torch.float32/64"
            raise TypeError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.a = Parameter(
            torch.empty(
                out_channels, in_channels, kernel_size, device=device, dtype=dtype
            )
        )
        self.b = Parameter(
            torch.empty(
                out_channels, in_channels, kernel_size, device=device, dtype=dtype
            )
        )
        if bias:
            self.bias_re = Parameter(
                torch.empty(out_channels, device=device, dtype=dtype)
            )
            self.bias_im = Parameter(
                torch.empty(out_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias_re", None)
            self.register_parameter("bias_im", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match `complex_xavier_uniform_` rectangular convention: identical
        # variance in `a` and `b` so that the effective complex weight has
        # the same fan-in / fan-out scaling as `ComplexConv1d`.
        fan_in = self.in_channels * self.kernel_size
        bound = (1.0 / fan_in) ** 0.5
        nn.init.uniform_(self.a, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)
        if self.bias_re is not None:
            nn.init.zeros_(self.bias_re)
            nn.init.zeros_(self.bias_im)

    def _build_real_weight(self) -> Tensor:
        # Construct (2*out, 2*in, K) real kernel from (out, in, K) (a, b).
        top = torch.cat([self.a, -self.b], dim=1)
        bottom = torch.cat([self.b, self.a], dim=1)
        return torch.cat([top, bottom], dim=0)

    def _build_real_bias(self) -> Tensor | None:
        if self.bias_re is None:
            return None
        return torch.cat([self.bias_re, self.bias_im])

    def forward(self, input: Tensor) -> Tensor:
        weight = self._build_real_weight()
        bias = self._build_real_bias()
        return F.conv1d(
            input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"bias={self.bias_re is not None}, "
            f"trainable_params_per_tap=2 (vs 4 for unconstrained Conv1d)"
        )


def complex_input_to_stacked(z: Tensor) -> Tensor:
    """Convert a complex `(B, C, L)` tensor into the `(B, 2C, L)` stacked-real
    layout used by `RotationEquivariantConv1d` and `RealStackedLinear`."""
    if not torch.is_complex(z):
        msg = "complex_input_to_stacked requires a complex tensor"
        raise TypeError(msg)
    return torch.cat([z.real, z.imag], dim=1)


def stacked_to_complex_output(x: Tensor) -> Tensor:
    """Inverse of `complex_input_to_stacked`: split a `(B, 2C, L)` real tensor
    back into a complex `(B, C, L)` tensor with the same channel ordering."""
    if torch.is_complex(x):
        msg = "stacked_to_complex_output requires a real tensor"
        raise TypeError(msg)
    c = x.shape[1] // 2
    return torch.complex(x[:, :c], x[:, c:])


def materialize_from_complex(
    complex_layer: ComplexConv1d,
) -> RotationEquivariantConv1d:
    """Build a `RotationEquivariantConv1d` whose forward pass equals
    `complex_layer` modulo the real/complex tensor convention. Used as a
    correctness witness for Proposition 3."""
    has_bias = complex_layer.bias is not None
    rot = RotationEquivariantConv1d(
        in_channels=complex_layer.in_channels,
        out_channels=complex_layer.out_channels,
        kernel_size=complex_layer.kernel_size,
        stride=complex_layer.stride,
        padding=complex_layer.padding,
        dilation=complex_layer.dilation,
        bias=has_bias,
        dtype=complex_layer.weight.real.dtype,
    )
    with torch.no_grad():
        rot.a.copy_(complex_layer.weight.real)
        rot.b.copy_(complex_layer.weight.imag)
        if has_bias:
            assert complex_layer.bias is not None
            assert rot.bias_re is not None and rot.bias_im is not None
            rot.bias_re.copy_(complex_layer.bias.real)
            rot.bias_im.copy_(complex_layer.bias.imag)
    return rot


__all__ = [
    "RotationEquivariantConv1d",
    "complex_input_to_stacked",
    "materialize_from_complex",
    "stacked_to_complex_output",
]
