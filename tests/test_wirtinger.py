from __future__ import annotations

import pytest
import torch

from cvnn.optim import pytorch_complex_gradient, validate_pytorch_complex_autograd


def test_validate_pytorch_complex_autograd_passes_known_cases() -> None:
    checks = validate_pytorch_complex_autograd()

    assert {check.name for check in checks} == {
        "abs_square_sum",
        "complex_mse_mean",
        "imag_square_sum",
        "magnitude_mse_mean",
        "real_projection",
        "real_square_sum",
    }
    assert all(check.passed for check in checks)


def test_pytorch_complex_gradient_matches_abs_square() -> None:
    z = torch.tensor([1 + 2j, -3 + 4j], dtype=torch.complex64)

    gradient = pytorch_complex_gradient(lambda value: value.abs().square().sum(), z)

    assert torch.allclose(gradient, 2 * z)


def test_pytorch_complex_gradient_rejects_real_tensor() -> None:
    with pytest.raises(TypeError):
        pytorch_complex_gradient(lambda value: value.square().sum(), torch.ones(2))
