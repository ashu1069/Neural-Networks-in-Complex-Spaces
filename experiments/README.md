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
