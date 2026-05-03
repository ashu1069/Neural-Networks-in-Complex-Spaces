# Experiments

Paper-track experiment code will live here.

Initial priorities:

1. Synthetic phase classification and controlled complex regression.
2. RF modulation classification once dataset access is settled.

Every completed run should write a result manifest to `results/` or to a
configured output directory with the same schema.

## Synthetic

Phase 3 includes a tiny complex linear regression convergence check:

```bash
uv run python experiments/synthetic/complex_linear_regression.py
```

To write a result manifest:

```bash
uv run python experiments/synthetic/complex_linear_regression.py \
  --output results/synthetic_complex_linear_regression.local.json
```

Phase 4 adds a phase-classification benchmark with these model families:

- `complex`: complex MLP over native complex inputs.
- `real_stacked`: real MLP over stacked `(real, imag)` inputs with the same
  hidden width as the complex model.
- `real_matched_params`: real MLP sized to match the complex model's scalar
  parameter count.
- `real_matched_flops`: real MLP sized to match the complex model's estimated
  forward multiply-add count.

Run the default CPU benchmark:

```bash
uv run python experiments/synthetic/phase_classification.py
```

For a quick smoke run:

```bash
uv run python experiments/synthetic/phase_classification.py \
  --seeds 0 \
  --n-train 128 \
  --n-test 128 \
  --steps 80 \
  --output-dir results/synthetic_phase_classification_smoke
```

Each run writes:

- `raw_runs.json`: one row per seed and model family.
- `summary.json`: aggregate mean/std/bootstrap confidence intervals.
- `summary.md`: paper-table-style markdown.
- `manifest.json`: environment, config, seed, metric, and artifact metadata.
