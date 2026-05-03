"""Validation helpers for PyTorch complex autograd conventions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from cvnn.losses import complex_mse_loss, magnitude_mse_loss

RealLoss = Callable[[Tensor], Tensor]
AnalyticGradient = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class AutogradConventionCheck:
    """One analytic-vs-PyTorch complex gradient comparison."""

    name: str
    max_abs_error: float
    passed: bool

    def to_dict(self) -> dict[str, float | bool | str]:
        return asdict(self)


def pytorch_complex_gradient(loss_fn: RealLoss, z: Tensor) -> Tensor:
    """Return PyTorch's complex gradient for a real-valued loss."""

    if not torch.is_complex(z):
        msg = "complex autograd checks require a complex tensor"
        raise TypeError(msg)
    variable = z.detach().clone().requires_grad_(True)
    loss = loss_fn(variable)
    if loss.dim() != 0:
        msg = "loss_fn must return a scalar tensor"
        raise ValueError(msg)
    loss.backward()  # type: ignore[no-untyped-call]
    if variable.grad is None:
        msg = "PyTorch did not populate a gradient"
        raise RuntimeError(msg)
    return variable.grad


def validate_pytorch_complex_autograd(
    *,
    dtype: torch.dtype = torch.complex128,
    atol: float = 1e-10,
) -> list[AutogradConventionCheck]:
    """Validate PyTorch gradients against hand-derived real-coordinate forms.

    For a real scalar objective `L(x, y)` and `z = x + i y`, PyTorch stores the
    steepest-descent real-coordinate gradient as `dL/dx + i dL/dy`.
    """

    z = torch.tensor(
        [1.0 + 2.0j, -0.5 + 0.75j, 0.25 - 1.25j],
        dtype=dtype,
    )
    target = torch.tensor(
        [-0.25 + 0.5j, 1.5 - 0.25j, -0.75 - 0.5j],
        dtype=dtype,
    )
    projection = torch.tensor(
        [0.5 - 1.0j, -1.25 + 0.5j, 0.25 + 0.75j],
        dtype=dtype,
    )
    cases: list[tuple[str, RealLoss, AnalyticGradient]] = [
        (
            "abs_square_sum",
            lambda value: value.abs().square().sum(),
            lambda value: 2 * value,
        ),
        (
            "real_square_sum",
            lambda value: value.real.square().sum(),
            lambda value: torch.complex(2 * value.real, torch.zeros_like(value.real)),
        ),
        (
            "imag_square_sum",
            lambda value: value.imag.square().sum(),
            lambda value: torch.complex(torch.zeros_like(value.imag), 2 * value.imag),
        ),
        (
            "real_projection",
            lambda value: torch.real(torch.conj(projection) * value).sum(),
            lambda _value: projection,
        ),
        (
            "complex_mse_mean",
            lambda value: complex_mse_loss(value, target),
            lambda value: 2 * (value - target) / value.numel(),
        ),
        (
            "magnitude_mse_mean",
            lambda value: magnitude_mse_loss(value, target),
            lambda value: (
                2.0 * (value.abs() - target.abs()) * value / value.abs() / value.numel()
            ),
        ),
    ]

    checks: list[AutogradConventionCheck] = []
    for name, loss_fn, analytic_gradient in cases:
        pytorch_gradient = pytorch_complex_gradient(loss_fn, z)
        expected = analytic_gradient(z)
        max_abs_error = float((pytorch_gradient - expected).abs().max().item())
        checks.append(
            AutogradConventionCheck(
                name=name,
                max_abs_error=max_abs_error,
                passed=max_abs_error <= atol,
            )
        )
    return checks


__all__ = [
    "AutogradConventionCheck",
    "validate_pytorch_complex_autograd",
    "pytorch_complex_gradient",
]
