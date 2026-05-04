from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.rf.gradient_telemetry import (
    TelemetryConfig,
    run_instrumented_training,
    telemetry_config_from_sweep_summary,
    write_telemetry_jsonl,
)


def _write_tiny_radioml(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    rows: list[np.ndarray] = []
    one_hot_rows: list[np.ndarray] = []
    snr_rows: list[np.ndarray] = []
    from experiments.rf.radioml import RADIOML_2018_01A_MODULATIONS

    archive_index = {n: i for i, n in enumerate(RADIOML_2018_01A_MODULATIONS)}
    for mod in ["BPSK", "QPSK"]:
        for snr in [0, 10]:
            for _ in range(8):
                rows.append(rng.standard_normal((128, 2)).astype(np.float32))
                one_hot = np.zeros(len(RADIOML_2018_01A_MODULATIONS), dtype=np.int8)
                one_hot[archive_index[mod]] = 1
                one_hot_rows.append(one_hot)
                snr_rows.append(np.array([[snr]], dtype=np.int8).reshape(1))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.stack(rows))
        handle.create_dataset("Y", data=np.stack(one_hot_rows))
        handle.create_dataset("Z", data=np.stack(snr_rows))


def _write_minimal_summary(path: Path, *, activation: str) -> None:
    payload = {
        "config": {
            "experiment": "radioml_modulation_sweep",
            "architecture": "conv",
            "kernel_size": 3,
            "modulations": ["BPSK", "QPSK"],
            "snr_db_levels": [0, 10],
            "max_per_class_per_snr": 4,
            "sample_length": 32,
            "val_fraction": 0.25,
            "activation": activation,
            "real_activation": "relu",
            "dtype": "complex64",
        },
        "matched_selections": [
            {
                "family": "complex",
                "selected_trial_index": 0,
                "selected_val_accuracy_mean": 0.8,
                "selected_test_accuracy_mean": 0.75,
                "selected_test_accuracy_std": 0.01,
                "selected_hyperparameters": {
                    "learning_rate": 0.01,
                    "hidden_features": 8,
                    "steps": 10,
                    "batch_size": 8,
                },
                "selected_extra": {"parameter_count_mean": 1000.0},
            },
            {
                "family": "real_matched_params",
                "selected_trial_index": 0,
                "selected_val_accuracy_mean": 0.7,
                "selected_test_accuracy_mean": 0.7,
                "selected_test_accuracy_std": 0.02,
                "selected_hyperparameters": {
                    "learning_rate": 0.01,
                    "hidden_features": 8,
                    "steps": 10,
                    "batch_size": 8,
                },
                "selected_extra": {"parameter_count_mean": 980.0},
            },
        ],
    }
    path.write_text(json.dumps(payload))


def test_telemetry_config_round_trip(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _write_minimal_summary(summary_path, activation="crelu")

    config = telemetry_config_from_sweep_summary(
        summary_path,
        activation="crelu",
        family="complex",
        seed=0,
        data_path=tmp_path / "tiny.hdf5",
    )

    assert isinstance(config, TelemetryConfig)
    assert config.activation == "crelu"
    assert config.family == "complex"
    assert config.seed == 0
    assert config.hyperparameters["learning_rate"] == 0.01
    assert config.architecture == "conv"


def test_telemetry_config_rejects_missing_family(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _write_minimal_summary(summary_path, activation="crelu")

    with pytest.raises(ValueError, match="not in"):
        telemetry_config_from_sweep_summary(
            summary_path,
            activation="crelu",
            family="real_stacked",  # absent from this minimal summary
            seed=0,
            data_path=tmp_path / "tiny.hdf5",
        )


def test_run_instrumented_training_logs_per_step(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    summary_path = tmp_path / "summary.json"
    _write_minimal_summary(summary_path, activation="crelu")
    archive = tmp_path / "tiny.hdf5"
    _write_tiny_radioml(archive)

    config = telemetry_config_from_sweep_summary(
        summary_path,
        activation="crelu",
        family="complex",
        seed=0,
        data_path=archive,
    )
    records, summary = run_instrumented_training(config, log_every_n=1)

    assert len(records) == config.hyperparameters["steps"]
    assert all(record.train_loss == record.train_loss for record in records)  # not NaN
    assert all(record.total_grad_norm >= 0.0 for record in records)
    assert summary["activation"] == "crelu"
    assert summary["family"] == "complex"
    assert 0.0 <= summary["test_accuracy"] <= 1.0
    # Per-layer keys should cover at least the conv weights
    assert any(
        "conv" in key for record in records for key in record.per_layer_grad_norm
    )


def test_telemetry_runs_against_synthetic_data_source(tmp_path: Path) -> None:
    """Synthetic data source should not require an HDF5 archive."""

    summary_path = tmp_path / "summary.json"
    payload = {
        "config": {
            "experiment": "rf_synthetic_modulation_sweep",
            "architecture": "conv",
            "kernel_size": 3,
            "modulations": ["bpsk", "qpsk"],
            "snr_db_levels": [0, 10],
            "n_per_class_per_snr": 4,
            "sample_length": 32,
            "val_fraction": 0.25,
            "activation": "crelu",
            "real_activation": "relu",
            "dtype": "complex64",
        },
        "matched_selections": [
            {
                "family": "complex",
                "selected_trial_index": 0,
                "selected_val_accuracy_mean": 0.8,
                "selected_test_accuracy_mean": 0.75,
                "selected_test_accuracy_std": 0.01,
                "selected_hyperparameters": {
                    "learning_rate": 0.01,
                    "hidden_features": 8,
                    "steps": 5,
                    "batch_size": 8,
                },
                "selected_extra": {"parameter_count_mean": 1000.0},
            },
        ],
    }
    summary_path.write_text(json.dumps(payload))

    config = telemetry_config_from_sweep_summary(
        summary_path,
        activation="crelu",
        family="complex",
        seed=0,
    )
    assert config.data_source == "synthetic"
    assert config.data_path is None

    records, summary = run_instrumented_training(config, log_every_n=1)
    assert len(records) == config.hyperparameters["steps"]
    assert 0.0 <= summary["test_accuracy"] <= 1.0


def test_write_telemetry_jsonl_round_trips(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    summary_path = tmp_path / "summary.json"
    _write_minimal_summary(summary_path, activation="crelu")
    archive = tmp_path / "tiny.hdf5"
    _write_tiny_radioml(archive)

    config = telemetry_config_from_sweep_summary(
        summary_path,
        activation="crelu",
        family="complex",
        seed=0,
        data_path=archive,
    )
    records, summary = run_instrumented_training(config, log_every_n=2)

    out = tmp_path / "log.jsonl"
    write_telemetry_jsonl(out, records=records, summary=summary)
    lines = out.read_text().splitlines()
    assert json.loads(lines[0]) == {"summary": summary}
    assert len(lines) == 1 + len(records)
