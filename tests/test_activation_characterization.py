from __future__ import annotations

import json
import subprocess
import sys

import torch

from cvnn.activations import complex_tanh
from cvnn.analysis import (
    ActivationSpec,
    activation_jacobian_norms,
    activation_specs,
    cauchy_riemann_residual,
    characterize_activation,
    complex_grid,
)


def test_activation_specs_cover_phase_2_set() -> None:
    names = {spec.name for spec in activation_specs()}

    assert names == {
        "complex_cardioid",
        "complex_tanh",
        "crelu",
        "modrelu",
        "siglog",
        "zrelu",
    }


def test_cauchy_riemann_residual_is_small_for_tanh_near_origin() -> None:
    z, x_values, y_values = complex_grid(grid_size=41, extent=0.5)
    residual = cauchy_riemann_residual(
        complex_tanh(z),
        dx=float(x_values[1] - x_values[0]),
        dy=float(y_values[1] - y_values[0]),
    )
    finite = residual[torch.isfinite(residual)]

    assert float(torch.quantile(finite, 0.95)) < 1e-3


def test_characterize_activation_returns_finite_summary() -> None:
    spec = ActivationSpec(
        name="identity",
        function=lambda z: z,
        module_factory=torch.nn.Identity,
        edge_definition="identity",
        singularity_notes="none",
    )

    summary = characterize_activation(
        spec,
        grid_size=17,
        extent=0.5,
        gradient_seeds=(0,),
    )

    assert summary.name == "identity"
    assert summary.finite_fraction == 1.0
    assert summary.cr_p95 < 1e-10
    assert summary.gradient_norm_mean > 0
    assert summary.jacobian_norm_mean > 0


def test_activation_jacobian_norms_isolated_from_mlp() -> None:
    norms = activation_jacobian_norms(lambda z: z, seeds=(0, 1, 2), n_samples=512)

    assert len(norms) == 3
    assert all(n > 0 for n in norms)
    assert all(torch.isfinite(torch.tensor(n)).item() for n in norms)


def test_activation_jacobian_norms_distinguishes_steep_from_flat() -> None:
    flat = activation_jacobian_norms(lambda z: 0.01 * z, seeds=(0, 1), n_samples=256)
    steep = activation_jacobian_norms(lambda z: 100.0 * z, seeds=(0, 1), n_samples=256)

    assert max(flat) < min(steep)


def test_characterization_script_writes_machine_readable_summary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    subprocess.run(
        [
            sys.executable,
            "scripts/characterize_activations.py",
            "--output-dir",
            str(tmp_path),
            "--grid-size",
            "17",
            "--gradient-seeds",
            "1",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert {row["name"] for row in comparison} == {
        "complex_cardioid",
        "complex_tanh",
        "crelu",
        "modrelu",
        "siglog",
        "zrelu",
    }
    assert all("jacobian_norm_mean" in row for row in comparison)
    assert (tmp_path / "comparison.md").exists()
