"""Loss functions for complex-valued experiments."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Reduction = Literal["none", "mean", "sum"]


def complex_mse_loss(
    input: Tensor,
    target: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Mean squared complex error: `|input - target|^2`."""

    _check_same_shape(input, target)
    loss = (input - target).abs().square()
    return _reduce(loss, reduction)


def magnitude_mse_loss(
    input: Tensor,
    target: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Mean squared error between complex magnitudes.

    Phase-blind by design: `1+0j` and `0+1j` both produce zero loss against
    a target of `1+0j`. Use this only when the task ignores phase (e.g. MRI
    magnitude reconstruction). Combine with `phase_aware_loss` if both
    magnitude and phase matter.
    """

    _check_same_shape(input, target)
    loss = (input.abs() - target.abs()).square()
    return _reduce(loss, reduction)


def phase_aware_loss(
    input: Tensor,
    target: Tensor,
    *,
    phase_weight: float = 1.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Complex MSE plus a phase-difference penalty.

    The phase term follows the Phase 3 plan:

    `|target - input|^2 + phase_weight * (1 - cos(arg(target) - arg(input)))`

    The squared error already implicitly weights phase via
    `|a - b|^2 = 2|a|^2(1 - cos(arg a - arg b))` when `|a| = |b|`. The
    explicit phase term adds a *magnitude-independent* penalty so phase
    accuracy does not vanish when both `|input|` and `|target|` are small.

    Gradient hazard at the origin: `torch.angle(0)` is defined as 0, but the
    *gradient* of `angle` at the origin is undefined. Predictions that pass
    through (or near) zero during training will inject noisy gradients
    through the phase term. If your task can produce near-zero predictions,
    add a small floor to `|input|` before this loss or use
    `magnitude_mse_loss` + `complex_mse_loss` separately.
    """

    if phase_weight < 0:
        msg = f"phase_weight must be non-negative, got {phase_weight}"
        raise ValueError(msg)
    _check_same_shape(input, target)
    squared_error = (input - target).abs().square()
    phase_difference = torch.angle(target) - torch.angle(input)
    phase_penalty = 1.0 - torch.cos(phase_difference)
    return _reduce(squared_error + phase_weight * phase_penalty, reduction)


def _check_same_shape(input: Tensor, target: Tensor) -> None:
    if input.shape != target.shape:
        msg = (
            f"input and target shapes must match, got {input.shape} and {target.shape}"
        )
        raise ValueError(msg)


def _reduce(loss: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    msg = f"unsupported reduction: {reduction}"
    raise ValueError(msg)


__all__ = [
    "Reduction",
    "complex_mse_loss",
    "magnitude_mse_loss",
    "phase_aware_loss",
]
