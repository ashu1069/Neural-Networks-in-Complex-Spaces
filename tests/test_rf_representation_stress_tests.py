from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from experiments.rf.representation_stress_tests import (
    InputTransform,
    apply_input_transform,
    build_stress_conditions,
)


def test_rf_stress_suite_declares_expected_conditions() -> None:
    condition_ids = {condition.condition_id for condition in build_stress_conditions()}

    assert "psk_representation" in condition_ids
    assert "qam_representation" in condition_ids
    assert "mixed_representation" in condition_ids
    assert "fixed_rotation_psk" in condition_ids
    assert "rotation_augmented_psk" in condition_ids
    assert "activation_zrelu" in condition_ids


def test_rf_stress_input_transforms_are_deterministic() -> None:
    inputs = torch.tensor([[3.0 + 4.0j, 1.0 + 0.0j]], dtype=torch.complex64)

    rotated = apply_input_transform(
        inputs,
        InputTransform("fixed_rotation", math.pi / 2.0),
        seed=0,
    )
    unit_magnitude = apply_input_transform(
        inputs,
        InputTransform("unit_magnitude"),
        seed=0,
    )
    random_left = apply_input_transform(
        inputs,
        InputTransform("random_rotation"),
        seed=123,
    )
    random_right = apply_input_transform(
        inputs,
        InputTransform("random_rotation"),
        seed=123,
    )

    assert torch.allclose(rotated, inputs * 1.0j, atol=1e-6)
    assert torch.allclose(unit_magnitude.abs(), torch.ones_like(inputs.abs()))
    assert torch.equal(random_left, random_right)


def test_rf_stress_script_writes_index_and_condition_outputs(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "experiments/rf/representation_stress_tests.py",
            "--preset",
            "smoke",
            "--tests",
            "psk_representation",
            "fixed_rotation_psk",
            "--seeds",
            "0",
            "--n-per-class-per-snr",
            "8",
            "--sample-length",
            "16",
            "--hidden-features",
            "4",
            "--steps",
            "5",
            "--batch-size",
            "16",
            "--bootstrap-samples",
            "20",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    index = json.loads((tmp_path / "index.json").read_text())

    assert (tmp_path / "index.md").exists()
    assert len(index["results"]) == 2
    assert (tmp_path / "psk_representation" / "summary.md").exists()
    assert (tmp_path / "fixed_rotation_psk" / "summary.json").exists()
