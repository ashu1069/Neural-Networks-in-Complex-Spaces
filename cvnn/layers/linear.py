"""Complex-valued linear layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from cvnn.init import ComplexDistribution, complex_xavier_uniform_, complex_zeros_


def complex_linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Apply a complex linear transform using matmul, not `F.linear`.

    MPS currently rejects native complex `torch.nn.functional.linear` on this
    project setup, while complex matmul works. Keeping the operation explicit
    gives us the same semantics and a useful accelerator path.

    The weight is applied via plain transpose (`x @ W.T`), not Hermitian
    (`x @ W.conj().T`). This matches `nn.Linear` semantics extended to `ℂ` and
    Trabelsi et al.'s "Deep Complex Networks" convention; it is what makes
    `ComplexLinear` reduce to `nn.Linear` when all imaginary parts are zero.
    Layers with a unitary or inner-product structure (e.g. unitary RNNs) need
    Hermitian transpose and should not reuse this helper.
    """

    output = input.matmul(weight.transpose(-2, -1))
    if bias is not None:
        output = output + bias
    return output


class ComplexLinear(nn.Module):
    """A linear layer for complex-valued tensors."""

    in_features: int
    out_features: int
    weight: Parameter
    bias: Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        init_distribution: ComplexDistribution = "rectangular",
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            msg = "in_features and out_features must be positive"
            raise ValueError(msg)

        dtype = _resolve_complex_dtype(dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.init_distribution = init_distribution
        self.weight = Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        complex_xavier_uniform_(self.weight, distribution=self.init_distribution)
        if self.bias is not None:
            complex_zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return complex_linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _resolve_complex_dtype(dtype: torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return torch.complex64
    if not dtype.is_complex:
        msg = f"ComplexLinear requires a complex dtype, got {dtype}"
        raise TypeError(msg)
    return dtype


__all__ = ["ComplexLinear", "complex_linear"]
