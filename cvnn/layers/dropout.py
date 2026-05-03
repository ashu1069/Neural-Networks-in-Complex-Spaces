"""Dropout layers for complex-valued tensors."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ComplexDropout(nn.Module):
    """Apply one real Bernoulli mask to both real and imaginary parts."""

    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if p < 0.0 or p > 1.0:
            msg = f"dropout probability has to be between 0 and 1, got {p}"
            raise ValueError(msg)
        self.p = float(p)
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        if self.p == 1.0:
            return input.zero_() if self.inplace else torch.zeros_like(input)

        keep_probability = 1.0 - self.p
        mask = torch.empty_like(input.real).bernoulli_(keep_probability)
        mask = mask.div_(keep_probability)
        if self.inplace:
            return input.mul_(mask)
        return input * mask

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


__all__ = ["ComplexDropout"]
