from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from experiments.rf import (
    RFBenchmarkConfig,
    make_synthetic_rf_modulation_dataset,
    run_rf_modulation_benchmark,
    train_rf_classifier,
)


def test_rf_dataset_is_deterministic() -> None:
    left = make_synthetic_rf_modulation_dataset(
        seed=3,
        modulations=("bpsk", "qpsk"),
        snr_db_levels=(0, 10),
        n_per_class_per_snr=8,
        sample_length=16,
    )
    right = make_synthetic_rf_modulation_dataset(
        seed=3,
        modulations=("bpsk", "qpsk"),
        snr_db_levels=(0, 10),
        n_per_class_per_snr=8,
        sample_length=16,
    )

    assert torch.equal(left.train_inputs, right.train_inputs)
    assert torch.equal(left.train_labels, right.train_labels)
    assert torch.equal(left.train_snr_db, right.train_snr_db)
    assert torch.equal(left.test_inputs, right.test_inputs)
    assert torch.equal(left.test_labels, right.test_labels)
    assert torch.equal(left.test_snr_db, right.test_snr_db)


def test_rf_dataset_shapes_and_class_balance() -> None:
    data = make_synthetic_rf_modulation_dataset(
        seed=0,
        modulations=("bpsk", "qpsk", "8psk"),
        snr_db_levels=(-10, 0, 10),
        n_per_class_per_snr=20,
        sample_length=32,
        train_fraction=0.75,
    )

    assert data.train_inputs.shape[1] == 32
    assert data.train_inputs.dtype == torch.complex64
    assert data.train_inputs.shape[0] == 3 * 3 * 15
    assert data.test_inputs.shape[0] == 3 * 3 * 5
    assert torch.equal(
        torch.bincount(data.train_labels, minlength=3),
        torch.full((3,), 45),
    )
    assert torch.equal(
        torch.bincount(data.test_labels, minlength=3),
        torch.full((3,), 15),
    )


def test_rf_dataset_rejects_invalid_args() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_synthetic_rf_modulation_dataset(modulations=())
    with pytest.raises(ValueError):
        make_synthetic_rf_modulation_dataset(snr_db_levels=())
    with pytest.raises(ValueError):
        make_synthetic_rf_modulation_dataset(n_per_class_per_snr=1)
    with pytest.raises(ValueError):
        make_synthetic_rf_modulation_dataset(sample_length=0)
    with pytest.raises(ValueError):
        make_synthetic_rf_modulation_dataset(train_fraction=0.0)
    with pytest.raises(ValueError):
        make_synthetic_rf_modulation_dataset(train_fraction=1.0)
    with pytest.raises(TypeError):
        make_synthetic_rf_modulation_dataset(dtype=torch.float32)


def test_matched_parameter_baseline_is_closer_than_naive_stacked() -> None:
    common_kwargs = dict(
        modulations=("bpsk", "qpsk", "8psk"),
        snr_db_levels=(0, 10),
        n_per_class_per_snr=8,
        sample_length=16,
        hidden_features=16,
        steps=20,
        batch_size=32,
    )
    complex_result = train_rf_classifier(
        seed=0,
        model_family="complex",
        **common_kwargs,  # type: ignore[arg-type]
    )
    matched_result = train_rf_classifier(
        seed=0,
        model_family="real_matched_params",
        **common_kwargs,  # type: ignore[arg-type]
    )
    stacked_result = train_rf_classifier(
        seed=0,
        model_family="real_stacked",
        **common_kwargs,  # type: ignore[arg-type]
    )

    matched_gap = abs(complex_result.parameter_count - matched_result.parameter_count)
    stacked_gap = abs(complex_result.parameter_count - stacked_result.parameter_count)
    assert matched_gap < stacked_gap


def test_rf_smoke_run_reports_per_snr_breakdown() -> None:
    config = RFBenchmarkConfig(
        seeds=(0,),
        model_families=("complex", "real_matched_params"),
        modulations=("bpsk", "qpsk"),
        snr_db_levels=(0, 10, 20),
        n_per_class_per_snr=32,
        sample_length=32,
        train_fraction=0.75,
        hidden_features=16,
        steps=50,
        batch_size=32,
        learning_rate=0.01,
        activation="crelu",
        real_activation="relu",
        architecture="mlp",
        kernel_size=7,
        device="cpu",
        dtype="complex64",
        bootstrap_samples=100,
        confidence=0.95,
    )

    runs, summaries = run_rf_modulation_benchmark(config)

    assert {run.model_family for run in runs} == {"complex", "real_matched_params"}
    for run in runs:
        assert set(run.accuracy_by_snr_db.keys()) == {"0", "10", "20"}
        assert 0.0 <= run.accuracy_by_snr_db["20"] <= 1.0
    for summary in summaries:
        assert set(summary.accuracy_by_snr_db_mean.keys()) == {"0", "10", "20"}


def test_rf_conv_architecture_runs() -> None:
    result = train_rf_classifier(
        seed=0,
        model_family="complex",
        modulations=("bpsk", "qpsk", "8psk"),
        snr_db_levels=(0, 10),
        n_per_class_per_snr=16,
        sample_length=32,
        hidden_features=8,
        steps=20,
        batch_size=32,
        architecture="conv",
        kernel_size=5,
    )

    assert result.parameter_count > 0
    assert 0.0 <= result.test_accuracy <= 1.0


def test_rf_classification_script_writes_evidence_files(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/rf/synthetic_modulation.py",
            "--seeds",
            "0",
            "--model-families",
            "complex",
            "real_matched_params",
            "--modulations",
            "bpsk",
            "qpsk",
            "--snr-db-levels",
            "0",
            "10",
            "--n-per-class-per-snr",
            "32",
            "--sample-length",
            "32",
            "--hidden-features",
            "16",
            "--steps",
            "40",
            "--batch-size",
            "32",
            "--bootstrap-samples",
            "100",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    raw_runs = json.loads((tmp_path / "raw_runs.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert "Synthetic RF Modulation Classification" in completed.stdout
    assert (tmp_path / "summary.md").exists()
    assert len(raw_runs) == 2
    assert len(summary["summaries"]) == 2
    assert manifest["metrics"]["summaries"] == summary["summaries"]
    assert manifest["dataset"]["name"] == "rf_synthetic_modulation"
