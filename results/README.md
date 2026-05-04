# Results

This directory stores intentionally committed example manifests and lightweight
paper evidence artifacts.

Large outputs, model checkpoints, raw datasets, and generated logs should stay
outside git unless explicitly needed for a small reproducibility fixture.

Phase 4 synthetic phase-classification runs write lightweight local artifacts to
`results/synthetic_phase_classification/` by default. Treat those as evidence
snapshots: keep them only when the config, environment, and git state are worth
preserving.

For RadioML activation ablations, use `radioml_modulation_sweep_crelu/` as the
corrected CReLU run. The older `radioml_modulation_sweep/` directory is retained
as a historical artifact from the odd-SNR request bug and is intentionally not
used by `scripts/generate_paper_figures.py`.
