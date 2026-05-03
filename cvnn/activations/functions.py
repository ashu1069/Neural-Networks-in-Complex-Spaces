"""Functional complex-valued activations."""

from __future__ import annotations

import torch
from torch import Tensor


def crelu(z: Tensor) -> Tensor:
    """Split ReLU: `ReLU(Re z) + i ReLU(Im z)`."""

    _check_complex(z, "crelu")
    return torch.complex(torch.relu(z.real), torch.relu(z.imag))


def zrelu(z: Tensor) -> Tensor:
    """Phase-gated ReLU that keeps values in the closed first quadrant."""

    _check_complex(z, "zrelu")
    mask = (z.real >= 0) & (z.imag >= 0)
    return torch.where(mask, z, torch.zeros_like(z))


def modrelu(z: Tensor, bias: float | Tensor, eps: float = 1e-12) -> Tensor:
    """Magnitude-gated ReLU that preserves phase and maps `z = 0` to zero.

    With `bias <= 0` the activation is continuous at the origin (a neighborhood
    of zero is zeroed out by `relu(|z| + bias)`). With `bias > 0` the limit as
    `|z| -> 0` is `relu(bias) * (z / |z|)`, which is direction-dependent, so
    the function is discontinuous at `z = 0` even though `modrelu(0) = 0` by
    construction. Arjovsky et al.'s original formulation assumes `bias <= 0`;
    if you allow the bias to be unconstrained during training, expect noisy
    gradients near the origin once it crosses zero.
    """

    _check_complex(z, "modrelu")
    magnitude = z.abs()
    bias_tensor = _as_real_tensor(bias, like=z)
    scale = torch.relu(magnitude + bias_tensor) / magnitude.clamp_min(eps)
    return scale * z


def complex_cardioid(z: Tensor) -> Tensor:
    """Cardioid activation: `0.5 * (1 + cos(arg z)) * z`."""

    _check_complex(z, "complex_cardioid")
    return 0.5 * (1.0 + torch.cos(torch.angle(z))) * z


def siglog(z: Tensor) -> Tensor:
    """Bounded smooth activation: `z / (1 + |z|)`."""

    _check_complex(z, "siglog")
    return z / (1.0 + z.abs())


def complex_tanh(z: Tensor) -> Tensor:
    """Complex hyperbolic tangent, a meromorphic cautionary baseline."""

    _check_complex(z, "complex_tanh")
    return torch.tanh(z)


def _check_complex(z: Tensor, name: str) -> None:
    if not torch.is_complex(z):
        msg = f"{name} expects a complex tensor"
        raise TypeError(msg)


def _as_real_tensor(value: float | Tensor, *, like: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=like.device, dtype=like.real.dtype)
    return torch.tensor(value, device=like.device, dtype=like.real.dtype)


__all__ = [
    "complex_cardioid",
    "complex_tanh",
    "crelu",
    "modrelu",
    "siglog",
    "zrelu",
]
