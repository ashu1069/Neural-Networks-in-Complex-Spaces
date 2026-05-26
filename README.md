# Neural Networks in Complex Spaces

Research code for complex-valued neural networks in PyTorch.

This repository contains:

- `cvnn/`: complex-valued layers, activations, initializers, baselines, and reproducibility helpers.
- `experiments/`: benchmark entry points for synthetic, RF, physics, and neuro-style tasks.
- `scripts/`: local support utilities.
- `tests/`: correctness and regression tests.

Datasets, generated results, notebook outputs, and paper build files are not committed.

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

Generated outputs default to `results/`, which is ignored. The RadioML dataset is not bundled; see [`docs/radioml.md`](docs/radioml.md) for local path configuration.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
