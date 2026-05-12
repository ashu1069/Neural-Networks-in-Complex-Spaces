"""Numerical-equivalence tests for `RotationEquivariantConv1d`.

These tests are the empirical face of Proposition 3 in the paper: a real
2-channel kernel commutes with $SO(2)$ iff it lies in the $aI+bJ$
subspace, which is the $\\mathbb{R}$-algebra image of $\\mathbb{C}$ under
the standard embedding. Operationally, the rotation-equivariant Conv1d
must reproduce a `ComplexConv1d`'s output (and gradients) bit-for-bit
modulo float rounding when initialized with the same effective complex
weights.
"""

from __future__ import annotations

import torch

from cvnn.baselines import (
    RotationEquivariantConv1d,
    complex_input_to_stacked,
    materialize_from_complex,
    stacked_to_complex_output,
)
from cvnn.layers import ComplexConv1d


def _seeded_complex_conv(seed: int = 0, *, bias: bool = True) -> ComplexConv1d:
    torch.manual_seed(seed)
    return ComplexConv1d(in_channels=3, out_channels=5, kernel_size=7, bias=bias)


def test_forward_matches_complex_conv1d():
    complex_layer = _seeded_complex_conv()
    rot_layer = materialize_from_complex(complex_layer)

    torch.manual_seed(42)
    z = torch.randn(4, 3, 32, dtype=torch.complex64)
    x = complex_input_to_stacked(z)

    y_complex = complex_layer(z)
    y_rot_real = rot_layer(x)
    y_rot = stacked_to_complex_output(y_rot_real)

    torch.testing.assert_close(y_rot, y_complex, atol=1e-6, rtol=1e-6)


def test_forward_matches_without_bias():
    complex_layer = _seeded_complex_conv(seed=1, bias=False)
    rot_layer = materialize_from_complex(complex_layer)
    torch.manual_seed(7)
    z = torch.randn(2, 3, 16, dtype=torch.complex64)
    y_rot = stacked_to_complex_output(rot_layer(complex_input_to_stacked(z)))
    torch.testing.assert_close(y_rot, complex_layer(z), atol=1e-6, rtol=1e-6)


def test_gradient_matches_complex_conv1d():
    """Check that gradients on (a, b, bias_re, bias_im) match the real and
    imaginary parts of the complex-layer gradients on (weight, bias).

    Both losses are normalized by the same total element count
    (`y_complex.numel()` --- number of complex outputs) so that
    `loss_rot.backward()` and `loss_complex.backward()` produce gradients
    on the same scale. PyTorch's complex autograd stores
    `weight.grad.real = $\\partial L / \\partial\\,\\mathrm{Re}(w)$`
    directly (the conjugate-Wirtinger convention is wrapped so the user
    sees real-coordinate gradients), so under matched normalization
    `rot.a.grad` should equal `complex.weight.grad.real` exactly.
    """
    complex_layer = _seeded_complex_conv(seed=2)
    rot_layer = materialize_from_complex(complex_layer)

    torch.manual_seed(13)
    z = torch.randn(8, 3, 24, dtype=torch.complex64)
    n_complex_outputs = float(
        z.shape[0]
        * complex_layer.out_channels
        * (z.shape[2] - complex_layer.kernel_size + 1)
    )

    y_complex = complex_layer(z)
    loss_complex = (
        y_complex.real.pow(2) + y_complex.imag.pow(2)
    ).sum() / n_complex_outputs
    loss_complex.backward()

    y_rot_real = rot_layer(complex_input_to_stacked(z))
    loss_rot = y_rot_real.pow(2).sum() / n_complex_outputs
    loss_rot.backward()

    assert complex_layer.weight.grad is not None
    assert complex_layer.bias is not None
    assert complex_layer.bias.grad is not None
    assert rot_layer.bias_re is not None and rot_layer.bias_im is not None
    assert rot_layer.bias_re.grad is not None
    assert rot_layer.bias_im.grad is not None

    torch.testing.assert_close(
        rot_layer.a.grad, complex_layer.weight.grad.real, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        rot_layer.b.grad, complex_layer.weight.grad.imag, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        rot_layer.bias_re.grad, complex_layer.bias.grad.real, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        rot_layer.bias_im.grad, complex_layer.bias.grad.imag, atol=1e-6, rtol=1e-5
    )


def test_parameter_count_is_half_of_unconstrained_2channel():
    """The whole point of Proposition 3: the equivariant subspace has 2
    real parameters per (i, j, k) tap, vs. 4 for an unconstrained Conv1d
    with 2*in / 2*out channels. The bias has 2*out entries in both."""
    in_c, out_c, k = 4, 6, 5
    rot = RotationEquivariantConv1d(in_c, out_c, k, bias=True)
    n_rot = sum(p.numel() for p in rot.parameters())

    unconstrained = torch.nn.Conv1d(2 * in_c, 2 * out_c, k, bias=True)
    n_unconstrained = sum(p.numel() for p in unconstrained.parameters())

    n_rot_kernel = 2 * in_c * out_c * k
    n_rot_bias = 2 * out_c
    n_unc_kernel = 2 * in_c * 2 * out_c * k
    n_unc_bias = 2 * out_c

    assert n_rot == n_rot_kernel + n_rot_bias
    assert n_unconstrained == n_unc_kernel + n_unc_bias
    assert n_rot_kernel * 2 == n_unc_kernel
