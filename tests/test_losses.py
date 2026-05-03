from __future__ import annotations

import pytest
import torch

from cvnn.losses import complex_mse_loss, magnitude_mse_loss, phase_aware_loss


def test_complex_mse_loss_values_and_reductions() -> None:
    input = torch.tensor([1 + 2j, 3 + 4j])
    target = torch.tensor([1 + 1j, 1 + 4j])

    unreduced = complex_mse_loss(input, target, reduction="none")

    assert torch.allclose(unreduced, torch.tensor([1.0, 4.0]))
    assert torch.allclose(complex_mse_loss(input, target), torch.tensor(2.5))
    assert torch.allclose(
        complex_mse_loss(input, target, reduction="sum"), torch.tensor(5.0)
    )


def test_magnitude_mse_loss_values() -> None:
    input = torch.tensor([3 + 4j, 0 + 0j])
    target = torch.tensor([0 + 0j, 0 + 1j])

    loss = magnitude_mse_loss(input, target, reduction="none")

    assert torch.allclose(loss, torch.tensor([25.0, 1.0]))


def test_phase_aware_loss_values() -> None:
    input = torch.tensor([1 + 0j, 1 + 0j])
    target = torch.tensor([1 + 0j, 0 + 1j])

    loss = phase_aware_loss(input, target, phase_weight=2.0, reduction="none")

    assert torch.allclose(loss[0], torch.tensor(0.0))
    assert torch.allclose(loss[1], torch.tensor(4.0))


def test_losses_reject_shape_mismatch() -> None:
    input = torch.ones(2, dtype=torch.complex64)
    target = torch.ones(3, dtype=torch.complex64)

    with pytest.raises(ValueError):
        complex_mse_loss(input, target)


def test_phase_aware_loss_rejects_negative_phase_weight() -> None:
    input = torch.ones(2, dtype=torch.complex64)

    with pytest.raises(ValueError):
        phase_aware_loss(input, input, phase_weight=-1.0)


def test_phase_aware_loss_gradcheck_away_from_origin() -> None:
    from torch.autograd import gradcheck

    target = torch.tensor(
        [1.0 + 0.5j, -0.75 + 0.25j, 0.5 - 1.0j], dtype=torch.complex128
    )
    input_ = torch.tensor(
        [0.5 + 0.5j, -0.25 + 0.75j, 1.0 - 0.5j],
        dtype=torch.complex128,
        requires_grad=True,
    )

    def loss_fn(value: torch.Tensor) -> torch.Tensor:
        return phase_aware_loss(value, target, phase_weight=2.0)

    assert gradcheck(loss_fn, (input_,), eps=1e-6, atol=1e-4)
