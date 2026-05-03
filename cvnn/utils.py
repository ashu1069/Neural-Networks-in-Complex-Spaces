"""Tensor utilities for complex-valued experiments."""

from __future__ import annotations

import torch
from torch import Tensor


def to_complex(real: Tensor, imag: Tensor | None = None) -> Tensor:
    """Return a complex tensor from real/imaginary parts or a real-pair view.

    If `imag` is omitted, `real` must either already be complex or have a final
    dimension of length 2 containing real and imaginary components.
    """

    if imag is None:
        if torch.is_complex(real):
            return real
        if real.shape[-1:] != (2,):
            msg = "real-pair tensors must have a final dimension of length 2"
            raise ValueError(msg)
        return torch.view_as_complex(real.contiguous())

    if torch.is_complex(real) or torch.is_complex(imag):
        msg = "real and imag inputs must be real-valued tensors"
        raise TypeError(msg)
    if real.shape != imag.shape:
        msg = f"real and imag shapes must match, got {real.shape} and {imag.shape}"
        raise ValueError(msg)
    return torch.complex(real, imag)


def as_real_pair(z: Tensor) -> Tensor:
    """Return a final-dimension real/imaginary view of a complex tensor.

    Wraps `torch.view_as_real`, which requires `z` to have viewable strides.
    Outputs of some ops are not viewable; call `.contiguous()` first if you
    hit a stride error.
    """

    if not torch.is_complex(z):
        msg = "as_real_pair expects a complex tensor"
        raise TypeError(msg)
    return torch.view_as_real(z)


def real_imag(z: Tensor) -> tuple[Tensor, Tensor]:
    """Split a complex tensor into real and imaginary tensors."""

    if not torch.is_complex(z):
        msg = "real_imag expects a complex tensor"
        raise TypeError(msg)
    return z.real, z.imag


def magnitude_phase(z: Tensor) -> tuple[Tensor, Tensor]:
    """Split a complex tensor into magnitude and phase tensors.

    `torch.angle(0+0j)` returns 0, but the gradient of `angle` at the origin
    is undefined. Phase-gated activations (e.g. cardioid, zReLU) that consume
    this output should clamp `|z|` away from zero before backprop.
    """

    if not torch.is_complex(z):
        msg = "magnitude_phase expects a complex tensor"
        raise TypeError(msg)
    return z.abs(), torch.angle(z)
