"""RF modulation classification benchmarks.

This package will eventually host a real-data benchmark on RadioML 2018.01A.
Until that loader lands, the working benchmark is the synthetic IQ stand-in
in `synthetic_modulation.py`, which uses i.i.d. PSK/QAM symbols + AWGN. The
synthetic task exercises the full benchmark harness (per-SNR accuracy, all
four baseline families, manifest, tuning log) on a sequence-shaped task,
without depending on the RadioML download.
"""

from experiments.rf.synthetic_modulation import (
    RFBenchmarkConfig,
    RFModulationData,
    RFRunResult,
    RFSummary,
    make_synthetic_rf_modulation_dataset,
    run_rf_modulation_benchmark,
    summarize_rf_runs,
    train_rf_classifier,
    write_rf_benchmark_outputs,
)

__all__ = [
    "RFBenchmarkConfig",
    "RFModulationData",
    "RFRunResult",
    "RFSummary",
    "make_synthetic_rf_modulation_dataset",
    "run_rf_modulation_benchmark",
    "summarize_rf_runs",
    "train_rf_classifier",
    "write_rf_benchmark_outputs",
]
