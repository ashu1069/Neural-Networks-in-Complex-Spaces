# Results

This directory stores intentionally committed example manifests and lightweight
paper evidence artifacts.

Large outputs, model checkpoints, raw datasets, and generated logs should stay
outside git unless explicitly needed for a small reproducibility fixture.

Phase 4 synthetic phase-classification runs write lightweight local artifacts to
`results/synthetic_phase_classification/` by default. Treat those as evidence
snapshots: keep them only when the config, environment, and git state are worth
preserving.
