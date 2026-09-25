# Neural Networks in Complex Spaces

Research code for complex-valued neural networks in PyTorch.

This repository contains:

- `cvnn/`: complex-valued layers, activations, initializers, baselines, and reproducibility helpers.
- `experiments/`: benchmark entry points for synthetic, RF, physics, and neuro-style tasks.
- `scripts/`: local support utilities.
- `tests/`: correctness and regression tests.

`results/` holds the committed experiment outputs backing every number in the
paper. Datasets, notebook outputs, and paper build files are not committed.

## Setup

This project uses Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy cvnn scripts experiments
```

## Common Commands

Generate local activation-characterization reports:

```bash
uv run python scripts/characterize_activations.py
```

Run the synthetic phase-classification benchmark:

```bash
uv run python experiments/synthetic/phase_classification.py
```

Run the synthetic RF modulation benchmark:

```bash
uv run python experiments/rf/synthetic_modulation.py
```

Audit local complex tensor support:

```bash
uv run python scripts/check_torch_complex_support.py
```

New runs write to `results/` by default. The RadioML dataset is not bundled;
see [`docs/radioml.md`](docs/radioml.md) for acquisition and local path
configuration.

## Results and reproducibility

`results/` contains one directory per reported configuration (31 in total).
Each holds a `manifest.json` conforming to
[`docs/result_manifest.schema.json`](docs/result_manifest.schema.json), which
records the git commit and dirty-tree flag, Python / PyTorch / CUDA versions,
device and dtype, the full experiment configuration, the seed list and the
resulting metrics — alongside per-seed raw runs, aggregated summaries, sweep
trial records and plots. Sweep-restart `checkpoint.json` files are the one
exclusion; they are bookkeeping rather than results.

Every table and figure in the paper is regenerated from those manifests:

```bash
uv run python scripts/build_appendix_tables.py
uv run python scripts/plot_crossover_figure.py
```

Note on provenance: the runs were executed from a working tree that was still
dirty relative to the commit their manifests name, so manifests carry that
earlier hash together with `git_dirty: true`. The committed code is what
should be used to reproduce them.

## CUDA note

`uv sync` resolves to a PyTorch build for a recent CUDA runtime, which will
fall back to CPU if the host driver is older. To pin a build matching an
older driver, for example CUDA 12.x:

```bash
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
```

Each manifest's `environment` block records the build actually used for that
run.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
