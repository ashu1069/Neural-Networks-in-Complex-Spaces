"""Synthetic benchmark experiments."""

from experiments.synthetic.complex_linear_regression import (
    LinearRegressionResult,
    SyntheticLinearRegressionData,
    closed_form_complex_linear_regression,
    make_complex_linear_regression,
    train_complex_linear_regression,
)
from experiments.synthetic.phase_classification import (
    PhaseBenchmarkConfig,
    PhaseClassificationData,
    PhaseRunResult,
    PhaseSummary,
    bootstrap_mean_ci,
    make_phase_classification,
    run_phase_classification_benchmark,
    summarize_phase_runs,
    train_phase_classifier,
    write_phase_benchmark_outputs,
)

__all__ = [
    "LinearRegressionResult",
    "PhaseBenchmarkConfig",
    "PhaseClassificationData",
    "PhaseRunResult",
    "PhaseSummary",
    "SyntheticLinearRegressionData",
    "bootstrap_mean_ci",
    "closed_form_complex_linear_regression",
    "make_complex_linear_regression",
    "make_phase_classification",
    "run_phase_classification_benchmark",
    "summarize_phase_runs",
    "train_complex_linear_regression",
    "train_phase_classifier",
    "write_phase_benchmark_outputs",
]
