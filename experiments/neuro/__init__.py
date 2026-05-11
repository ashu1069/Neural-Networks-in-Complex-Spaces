"""Neuroscience pilot experiments for complex analytic signals."""

from experiments.neuro.eeg_analytic_signal import (
    EEGCondition,
    EEGRunConfig,
    build_eeg_conditions,
    make_amplitude_event_dataset,
    make_pac_dataset,
    make_phase_locking_dataset,
    run_condition,
)

__all__ = [
    "EEGCondition",
    "EEGRunConfig",
    "build_eeg_conditions",
    "make_amplitude_event_dataset",
    "make_pac_dataset",
    "make_phase_locking_dataset",
    "run_condition",
]
