"""Real-valued stacked-channel baseline layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cvnn.layers import ComplexLinear


class RealStackedLinear(nn.Module):
    """Real linear layer that treats `(real, imag)` as two stacked channels.

    Input shape: `(..., 2 * in_features)` with the first `in_features` entries
    holding real parts and the next `in_features` holding imaginary parts.
    Output shape: `(..., 2 * out_features)`, same convention.

    This is the "naive real baseline": the network has no complex semantics
    and learns an arbitrary real-valued mapping. It has roughly twice the
    real parameter count of an equivalent `ComplexLinear` (see
    `count_real_parameters`).
    """

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            msg = "in_features and out_features must be positive"
            raise ValueError(msg)
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(
            2 * in_features,
            2 * out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Tensor | None:
        return self.linear.bias

    def forward(self, input: Tensor) -> Tensor:
        output: Tensor = self.linear(input)
        return output


def complex_to_real_reparam(complex_linear: ComplexLinear) -> RealStackedLinear:
    """Build a `RealStackedLinear` with input-output behavior identical to a
    given `ComplexLinear`.

    Given complex weight `W = A + iB` (shape `(out, in)`) and bias
    `b = b_r + i b_i`, the equivalent real `2*in -> 2*out` block matrix is:

        [ A  -B ]      [ b_r ]
        [ B   A ]  ;   [ b_i ]

    so for input `(x, y)` (real and imag parts stacked along the last dim),
    the output equals `(Re(W(x+iy) + b), Im(W(x+iy) + b))`.

    Use this to confirm a complex model's behavior is *exactly* representable
    in real coordinates - a sanity check before claiming a parameter-matched
    real baseline is meaningfully different.
    """

    in_features = complex_linear.in_features
    out_features = complex_linear.out_features
    has_bias = complex_linear.bias is not None
    real_layer = RealStackedLinear(
        in_features,
        out_features,
        bias=has_bias,
        dtype=complex_linear.weight.real.dtype,
    )
    with torch.no_grad():
        real_part = complex_linear.weight.real
        imag_part = complex_linear.weight.imag
        top = torch.cat([real_part, -imag_part], dim=1)
        bottom = torch.cat([imag_part, real_part], dim=1)
        real_layer.linear.weight.copy_(torch.cat([top, bottom], dim=0))
        if has_bias:
            assert complex_linear.bias is not None
            assert real_layer.linear.bias is not None
            real_layer.linear.bias.copy_(
                torch.cat([complex_linear.bias.real, complex_linear.bias.imag])
            )
    return real_layer


def count_real_parameters(module: nn.Module) -> int:
    """Count real-valued parameters; complex parameters count as 2 each.

    Use this when matching parameter budgets across a complex model and a
    real-valued baseline. A complex tensor of `n` elements consumes `2n`
    real-valued slots, so a fair "matched parameters" baseline has its real
    `numel()` equal to this count.
    """

    total = 0
    for parameter in module.parameters():
        if torch.is_complex(parameter):
            total += 2 * parameter.numel()
        else:
            total += parameter.numel()
    return total


__all__ = [
    "RealStackedLinear",
    "complex_to_real_reparam",
    "count_real_parameters",
]
