from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from experiments.physics.quantum_wavefunction import (
    InputTransform,
    apply_input_transform,
    build_quantum_conditions,
    make_momentum_phase_dataset,
    make_potential_inverse_dataset,
)


def test_quantum_conditions_declared() -> None:
    condition_ids = {condition.condition_id for condition in build_quantum_conditions()}

    assert "momentum_phase" in condition_ids
    assert "potential_inverse" in condition_ids
    assert "global_phase_shift" in condition_ids
    assert "global_phase_augmented" in condition_ids


def test_momentum_phase_dataset_is_balanced_and_deterministic() -> None:
    left = make_momentum_phase_dataset(
        seed=7,
        examples_per_class=10,
        grid_size=32,
        train_fraction=0.6,
    )
    right = make_momentum_phase_dataset(
        seed=7,
        examples_per_class=10,
        grid_size=32,
        train_fraction=0.6,
    )

    assert torch.equal(left.train_inputs, right.train_inputs)
    assert torch.equal(left.test_inputs, right.test_inputs)
    assert left.train_inputs.shape == (24, 32)
    assert left.test_inputs.shape == (16, 32)
    assert torch.equal(torch.bincount(left.train_labels), torch.full((4,), 6))
    assert torch.equal(torch.bincount(left.test_labels), torch.full((4,), 4))


def test_potential_inverse_dataset_shapes() -> None:
    data = make_potential_inverse_dataset(
        seed=2,
        examples_per_class=8,
        grid_size=32,
        train_fraction=0.75,
        evolution_steps=3,
    )

    assert data.train_inputs.shape == (30, 32)
    assert data.test_inputs.shape == (10, 32)
    assert data.train_inputs.dtype == torch.complex64
    assert torch.equal(torch.bincount(data.train_labels), torch.full((5,), 6))
    assert torch.equal(torch.bincount(data.test_labels), torch.full((5,), 2))


def test_global_phase_transform_is_deterministic() -> None:
    inputs = torch.tensor([[1.0 + 0.0j, 0.0 + 1.0j]], dtype=torch.complex64)

    fixed = apply_input_transform(
        inputs,
        InputTransform("fixed_global_phase", math.pi / 2.0),
        seed=0,
    )
    random_left = apply_input_transform(
        inputs,
        InputTransform("random_global_phase"),
        seed=123,
    )
    random_right = apply_input_transform(
        inputs,
        InputTransform("random_global_phase"),
        seed=123,
    )

    assert torch.allclose(fixed, inputs * 1.0j, atol=1e-6)
    assert torch.equal(random_left, random_right)


def test_quantum_script_writes_outputs(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "experiments/physics/quantum_wavefunction.py",
            "--preset",
            "smoke",
            "--tests",
            "momentum_phase",
            "potential_inverse",
            "--seeds",
            "0",
            "--examples-per-class",
            "8",
            "--grid-size",
            "24",
            "--evolution-steps",
            "3",
            "--hidden-features",
            "4",
            "--train-steps",
            "3",
            "--batch-size",
            "16",
            "--bootstrap-samples",
            "20",
            "--no-progress",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    index = json.loads((tmp_path / "index.json").read_text())

    assert (tmp_path / "index.md").exists()
    assert (tmp_path / "accuracy_heatmap.png").exists()
    assert (tmp_path / "best_accuracy_by_condition.png").exists()
    assert len(index["results"]) == 2
    assert (tmp_path / "momentum_phase" / "summary.md").exists()
    assert (tmp_path / "momentum_phase" / "accuracy_bar.png").exists()
    assert (tmp_path / "momentum_phase" / "accuracy_by_class.png").exists()
    assert (tmp_path / "potential_inverse" / "summary.json").exists()
