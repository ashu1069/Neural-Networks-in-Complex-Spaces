# Experiments

The drivers that produced `results/`. The exact command for each paper result
is in the top-level [`README.md`](../README.md#reproducing-the-paper).

| Driver | What it runs |
|---|---|
| `rf/representation_stress_tests.py` | Synthetic RF stress tests (PSK, QAM, mixed, SNR, normalisation, rotation) across all baseline families; `--activation` overrides the complex activation. |
| `rf/sweep_radioml.py` | RadioML 2018.01A random-search sweep, reported under both selection rules; `--grad-clip-norm` for the clipping intervention. |
| `rf/radioml.py`, `rf/path_config.py` | RadioML loader and path resolution; see [`docs/radioml.md`](../docs/radioml.md). |
| `rf/synthetic_modulation.py` | Model families, including the $U(1)$-constrained `real_equivariant` arm, and the synthetic IQ generator shared by the RF drivers. |
| `rf/gradient_telemetry.py` | Per-step gradient telemetry harness. |
| `physics/quantum_wavefunction.py` | Quantum wavefunction pilot. |
| `neuro/eeg_analytic_signal.py` | EEG analytic-signal pilot. |
| `synthetic/phase_classification.py` | Shared classifier heads and bootstrap statistics used by the drivers above. |
| `_sweep.py` | Random search and the two selection rules, following [`docs/tuning_budget.md`](../docs/tuning_budget.md). |

Every run writes `manifest.json`, `raw_runs.json`, `summary.json` and
`summary.md`, and supports `--resume`.
