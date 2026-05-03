from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.rf.radioml import (
    RADIOML_2018_01A_MODULATIONS,
    RADIOML_2018_01A_SNR_DB,
    load_radioml_2018_01a,
)


def _write_synthetic_radioml(
    path: Path,
    *,
    modulations: list[str],
    snr_db_levels: list[int],
    n_per_class_per_snr: int,
    sample_length: int = 128,
    seed: int = 0,
) -> None:
    """Write a tiny RadioML-shaped HDF5 file for tests."""

    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(seed)
    n_classes_archive = len(RADIOML_2018_01A_MODULATIONS)

    rows: list[np.ndarray] = []
    one_hot_rows: list[np.ndarray] = []
    snr_rows: list[np.ndarray] = []
    archive_index = {name: idx for idx, name in enumerate(RADIOML_2018_01A_MODULATIONS)}
    for mod in modulations:
        for snr in snr_db_levels:
            for _ in range(n_per_class_per_snr):
                rows.append(rng.standard_normal((sample_length, 2)).astype(np.float32))
                one_hot = np.zeros(n_classes_archive, dtype=np.int8)
                one_hot[archive_index[mod]] = 1
                one_hot_rows.append(one_hot)
                snr_rows.append(np.array([[snr]], dtype=np.int8).reshape(1))

    x = np.stack(rows, axis=0)
    y = np.stack(one_hot_rows, axis=0)
    z = np.stack(snr_rows, axis=0)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=x)
        handle.create_dataset("Y", data=y)
        handle.create_dataset("Z", data=z)


def test_load_radioml_filters_modulations_and_snr(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    archive = tmp_path / "tiny.hdf5"
    _write_synthetic_radioml(
        archive,
        modulations=["BPSK", "QPSK", "8PSK", "16QAM"],
        snr_db_levels=[-10, 0, 10, 20],
        n_per_class_per_snr=8,
        sample_length=128,
    )

    data = load_radioml_2018_01a(
        archive,
        modulations=["BPSK", "QPSK"],
        snr_db_levels=[0, 10],
        sample_length=64,
    )

    assert data.modulation_names == ("BPSK", "QPSK")
    assert data.snr_db_levels == (0, 10)
    assert data.sample_length == 64
    assert data.train_inputs.shape[1] == 64
    assert data.train_inputs.dtype == torch.complex64
    # 2 mods × 2 snrs × 8 examples = 32 total; 80% train = 6 per bucket × 4 buckets = 24
    assert data.train_inputs.shape[0] == 24
    assert data.test_inputs.shape[0] == 8
    assert set(data.test_snr_db.tolist()) == {0, 10}
    assert set(data.test_labels.tolist()).issubset({0, 1})


def test_load_radioml_max_per_class_per_snr(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    archive = tmp_path / "tiny.hdf5"
    _write_synthetic_radioml(
        archive,
        modulations=["BPSK", "QPSK"],
        snr_db_levels=[0, 10],
        n_per_class_per_snr=20,
    )

    data = load_radioml_2018_01a(
        archive,
        modulations=["BPSK", "QPSK"],
        snr_db_levels=[0, 10],
        max_per_class_per_snr=5,
        sample_length=128,
    )

    # 2 mods × 2 snrs × 5 examples = 20 total; 80% train = 4 per bucket × 4 buckets
    assert data.train_inputs.shape[0] == 16
    assert data.test_inputs.shape[0] == 4


def test_load_radioml_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_radioml_2018_01a(tmp_path / "nope.hdf5")


def test_load_radioml_rejects_unknown_modulation(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    archive = tmp_path / "tiny.hdf5"
    _write_synthetic_radioml(
        archive,
        modulations=["BPSK"],
        snr_db_levels=[0],
        n_per_class_per_snr=4,
    )

    with pytest.raises(ValueError):
        load_radioml_2018_01a(archive, modulations=["NOT_A_REAL_MOD"])


def test_load_radioml_raises_when_no_examples_match(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    archive = tmp_path / "tiny.hdf5"
    _write_synthetic_radioml(
        archive,
        modulations=["BPSK"],
        snr_db_levels=[0],
        n_per_class_per_snr=4,
    )

    with pytest.raises(ValueError, match="no examples matched"):
        load_radioml_2018_01a(archive, modulations=["BPSK"], snr_db_levels=[15])


def test_radioml_constants_are_canonical_24_and_26() -> None:
    assert len(RADIOML_2018_01A_MODULATIONS) == 24
    assert len(RADIOML_2018_01A_SNR_DB) == 26
    assert RADIOML_2018_01A_SNR_DB[0] == -20
    assert RADIOML_2018_01A_SNR_DB[-1] == 30
