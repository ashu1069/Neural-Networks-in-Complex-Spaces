from __future__ import annotations

import pytest
import torch

from cvnn.utils import as_real_pair, magnitude_phase, real_imag, to_complex


def test_to_complex_from_real_and_imag() -> None:
    real = torch.tensor([1.0, 2.0])
    imag = torch.tensor([3.0, 4.0])

    z = to_complex(real, imag)

    assert z.dtype == torch.complex64
    assert torch.allclose(z.real, real)
    assert torch.allclose(z.imag, imag)


def test_to_complex_from_real_pair() -> None:
    pair = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    z = to_complex(pair)

    assert torch.allclose(z, torch.tensor([1.0 + 2.0j, 3.0 + 4.0j]))


def test_as_real_pair_round_trips() -> None:
    z = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])

    pair = as_real_pair(z)

    assert pair.shape == (2, 2)
    assert torch.allclose(to_complex(pair), z)


def test_splitters_require_complex_input() -> None:
    real = torch.ones(2)

    with pytest.raises(TypeError):
        as_real_pair(real)
    with pytest.raises(TypeError):
        real_imag(real)
    with pytest.raises(TypeError):
        magnitude_phase(real)


def test_magnitude_phase() -> None:
    z = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j])

    magnitude, phase = magnitude_phase(z)

    assert torch.allclose(magnitude, torch.ones(2))
    assert torch.allclose(phase, torch.tensor([0.0, torch.pi / 2]))
