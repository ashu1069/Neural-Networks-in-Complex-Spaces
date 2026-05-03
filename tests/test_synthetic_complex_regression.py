from __future__ import annotations

import json
import subprocess
import sys

import torch

from experiments.synthetic import (
    closed_form_complex_linear_regression,
    make_complex_linear_regression,
    train_complex_linear_regression,
)


def test_closed_form_solution_recovers_noise_free_generator() -> None:
    data = make_complex_linear_regression(seed=0)

    weight, bias = closed_form_complex_linear_regression(data.inputs, data.targets)

    assert torch.allclose(weight, data.true_weight, atol=1e-5, rtol=1e-5)
    assert torch.allclose(bias, data.true_bias, atol=1e-5, rtol=1e-5)


def test_stock_adamw_converges_to_closed_form_solution() -> None:
    result = train_complex_linear_regression(steps=200, learning_rate=0.05)

    assert result.final_loss < 1e-6
    assert result.prediction_mse_to_closed_form < 1e-6
    assert result.weight_mse_to_closed_form < 1e-6
    assert result.bias_mse_to_closed_form < 1e-6


def test_stock_sgd_converges_to_closed_form_solution() -> None:
    result = train_complex_linear_regression(
        optimizer="sgd", steps=400, learning_rate=0.05
    )

    assert result.final_loss < 1e-4
    assert result.prediction_mse_to_closed_form < 1e-4


def test_closed_form_handles_noisy_targets() -> None:
    data = make_complex_linear_regression(seed=0, noise_std=0.1)
    weight, bias = closed_form_complex_linear_regression(data.inputs, data.targets)

    weight_error = (weight - data.true_weight).abs().mean().item()
    assert weight_error > 1e-3
    assert weight_error < 0.5


def test_synthetic_regression_script_writes_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest_path = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/synthetic/complex_linear_regression.py",
            "--steps",
            "100",
            "--output",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    manifest = json.loads(manifest_path.read_text())
    assert result["final_loss"] < 1e-4
    assert manifest["metrics"]["final_loss"] == result["final_loss"]
