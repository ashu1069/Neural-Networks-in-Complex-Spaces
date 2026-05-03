"""RF modulation classification benchmarks.

The synthetic IQ stand-in in `synthetic_modulation.py` uses i.i.d. PSK/QAM
symbols + AWGN and remains the fast, bundled benchmark. `radioml.py` adds
local loading for RadioML 2018.01A, reusing the same benchmark harness once
the gated HDF5 archive is available under `data/`.
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
