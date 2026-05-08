from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from experiments.synthetic import (
    PhaseBenchmarkConfig,
    make_phase_classification,
    run_phase_classification_benchmark,
    train_phase_classifier,
)


def test_phase_classification_data_is_deterministic() -> None:
    left = make_phase_classification(seed=7, n_train=32, n_test=16)
    right = make_phase_classification(seed=7, n_train=32, n_test=16)

    assert torch.equal(left.train_inputs, right.train_inputs)
    assert torch.equal(left.train_labels, right.train_labels)
    assert torch.equal(left.test_inputs, right.test_inputs)
    assert torch.equal(left.test_labels, right.test_labels)


def test_phase_classification_labels_are_balanced() -> None:
    data = make_phase_classification(seed=0, n_train=64, n_test=64, n_classes=4)

    train_counts = torch.bincount(data.train_labels, minlength=4)
    test_counts = torch.bincount(data.test_labels, minlength=4)
    assert torch.equal(train_counts, torch.full((4,), 16))
    assert torch.equal(test_counts, torch.full((4,), 16))


def test_matched_parameter_baseline_matches_complex_scalar_count() -> None:
    complex_result = train_phase_classifier(
        seed=0,
        model_family="complex",
        n_train=96,
        n_test=96,
        steps=60,
    )
    matched_result = train_phase_classifier(
        seed=0,
        model_family="real_matched_params",
        n_train=96,
        n_test=96,
        steps=60,
    )

    assert complex_result.parameter_count == matched_result.parameter_count
    assert matched_result.hidden_features > complex_result.hidden_features


def test_phase_benchmark_smoke_run_reports_all_requested_families() -> None:
    config = PhaseBenchmarkConfig(
        seeds=(0,),
        model_families=("complex", "real_matched_params"),
        n_train=128,
        n_test=128,
        n_classes=4,
        hidden_features=16,
        steps=80,
        learning_rate=0.02,
        class_spread=None,
        noise_std=0.05,
        activation="crelu",
        real_activation="relu",
        device="cpu",
        dtype="complex64",
        bootstrap_samples=100,
        confidence=0.95,
    )

    runs, summaries = run_phase_classification_benchmark(config)

    assert {run.model_family for run in runs} == {"complex", "real_matched_params"}
    assert {summary.model_family for summary in summaries} == {
        "complex",
        "real_matched_params",
    }
    assert all(run.test_accuracy >= 0.95 for run in runs)


def test_phase_representation_ablation_separates_magnitude_from_phase() -> None:
    polar_result = train_phase_classifier(
        seed=0,
        model_family="real_polar",
        n_train=256,
        n_test=256,
        n_classes=4,
        hidden_features=16,
        steps=120,
        learning_rate=0.02,
        noise_std=0.05,
    )
    magnitude_result = train_phase_classifier(
        seed=0,
        model_family="real_magnitude",
        n_train=256,
        n_test=256,
        n_classes=4,
        hidden_features=16,
        steps=120,
        learning_rate=0.02,
        noise_std=0.05,
    )

    assert polar_result.test_accuracy >= 0.95
    assert magnitude_result.test_accuracy <= 0.5


def test_phase_classification_script_writes_evidence_files(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/synthetic/phase_classification.py",
            "--seeds",
            "0",
            "--model-families",
            "complex",
            "real_matched_params",
            "--n-train",
            "128",
            "--n-test",
            "128",
            "--steps",
            "80",
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

    assert "Synthetic Phase Classification" in completed.stdout
    assert (tmp_path / "summary.md").exists()
    assert len(raw_runs) == 2
    assert len(summary["summaries"]) == 2
    assert manifest["metrics"]["summaries"] == summary["summaries"]
