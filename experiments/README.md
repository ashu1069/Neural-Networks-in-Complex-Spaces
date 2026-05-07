# Experiments

Paper-track experiment code will live here.

Initial priorities:

1. Synthetic phase classification and controlled complex regression.
2. Synthetic RF modulation classification as a stand-in for RadioML.
3. Real RF modulation classification once dataset access is settled.

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

## Sweeps

Budgeted sweeps use shared hyperparameter samples across families. The primary
paper table uses the matched shared-trial comparison; independent family winners
are diagnostic only.

```bash
uv run python experiments/synthetic/sweep_phase_classification.py
uv run python experiments/rf/sweep_synthetic_modulation.py --device cuda
```

## RF

The paper-track RF results currently use synthetic IQ + AWGN:

```bash
uv run python experiments/rf/synthetic_modulation.py
```

The real RadioML 2018.01A loader expects the local archive at
`data/GOLD_XYZ_OSC.0001_1024.hdf5` and the fixed class-order sidecar next to it:

```bash
uv run python experiments/rf/sweep_radioml.py --device cuda
```

For Colab or mounted-drive runs, copy `config/radioml_paths.example.json` to
`config/radioml_paths.json`, set the absolute mounted-drive path there, and run
the same command without a long `--data-path` flag. CLI flags and the
`RADIOML_DATA_PATH` / `RADIOML_CLASSES_PATH` environment variables still
override the config.

All sweep scripts support `--resume`. They write `checkpoint.json` after each
completed seed run, plus `training_params.json`, `loss_curves_all.png`, and
`loss_curves_selected.png` after a completed sweep.

For H100/A100 runs with enough GPU memory, add `--cache-data-device device` to
RadioML sweeps so capped dataset tensors stay on GPU between trials.

The full RF sweep is GPU-oriented. CPU smoke runs should reduce
`--sample-length`, `--n-per-class-per-snr`, and/or `--n-trials`.
