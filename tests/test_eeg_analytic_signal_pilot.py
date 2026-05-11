from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from experiments.neuro.eeg_analytic_signal import (
    InputTransform,
    apply_input_transform,
    build_eeg_conditions,
    make_amplitude_event_dataset,
    make_pac_dataset,
    make_phase_locking_dataset,
)


def test_eeg_conditions_declared() -> None:
    condition_ids = {condition.condition_id for condition in build_eeg_conditions()}

    assert "phase_locking" in condition_ids
    assert "amplitude_event" in condition_ids
    assert "phase_amplitude_coupling" in condition_ids
    assert "reference_phase_shift" in condition_ids
    assert "reference_phase_augmented" in condition_ids


def test_phase_locking_dataset_is_balanced_and_deterministic() -> None:
    left = make_phase_locking_dataset(
        seed=7,
        examples_per_class=10,
        n_channels=4,
        time_steps=24,
        train_fraction=0.6,
    )
    right = make_phase_locking_dataset(
        seed=7,
        examples_per_class=10,
        n_channels=4,
        time_steps=24,
        train_fraction=0.6,
    )

    assert torch.equal(left.train_inputs, right.train_inputs)
    assert torch.equal(left.test_inputs, right.test_inputs)
    assert left.train_inputs.shape == (24, 96)
    assert left.test_inputs.shape == (16, 96)
    assert torch.equal(torch.bincount(left.train_labels), torch.full((4,), 6))
    assert torch.equal(torch.bincount(left.test_labels), torch.full((4,), 4))


def test_amplitude_and_pac_dataset_shapes() -> None:
    amplitude = make_amplitude_event_dataset(
        seed=2,
        examples_per_class=8,
        n_channels=4,
        time_steps=24,
        train_fraction=0.75,
    )
    pac = make_pac_dataset(
        seed=2,
        examples_per_class=8,
        n_channels=4,
        time_steps=24,
        train_fraction=0.75,
    )

    assert amplitude.train_inputs.shape == (24, 96)
    assert amplitude.test_inputs.shape == (8, 96)
    assert pac.train_inputs.shape == (24, 96)
    assert pac.test_inputs.shape == (8, 96)
    assert amplitude.train_inputs.dtype == torch.complex64
    assert pac.train_inputs.dtype == torch.complex64


def test_reference_phase_transform_is_deterministic() -> None:
    inputs = torch.tensor([[1.0 + 0.0j, 0.0 + 1.0j]], dtype=torch.complex64)

    fixed = apply_input_transform(
        inputs,
        InputTransform("fixed_reference_phase", math.pi / 2.0),
        seed=0,
    )
    random_left = apply_input_transform(
        inputs,
        InputTransform("random_reference_phase"),
        seed=123,
    )
    random_right = apply_input_transform(
        inputs,
        InputTransform("random_reference_phase"),
        seed=123,
    )

    assert torch.allclose(fixed, inputs * 1.0j, atol=1e-6)
    assert torch.equal(random_left, random_right)


def test_eeg_script_writes_outputs(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "experiments/neuro/eeg_analytic_signal.py",
            "--preset",
            "smoke",
            "--tests",
            "phase_locking",
            "amplitude_event",
            "--seeds",
            "0",
            "--examples-per-class",
            "8",
            "--time-steps",
            "24",
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
    assert (tmp_path / "phase_locking" / "summary.md").exists()
    assert (tmp_path / "phase_locking" / "accuracy_bar.png").exists()
    assert (tmp_path / "amplitude_event" / "summary.json").exists()
