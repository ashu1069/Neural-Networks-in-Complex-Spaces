# CUDA Notes

CUDA is the scale-up target for larger experiment sweeps and final confirmation
runs. Apple Silicon remains the local development environment, and CPU remains
the correctness reference.

## Server Setup

From a CUDA server:

```bash
uv sync --locked --all-groups
uv run python scripts/check_torch_complex_support.py --device cuda
```

Use `--fail-on-gap` in CI-like jobs when unsupported CUDA behavior should stop
the run:

```bash
uv run python scripts/check_torch_complex_support.py --device cuda --fail-on-gap
```

## Run Policy

- Run CPU smoke tests first.
- Run the support audit on the CUDA server before any large sweep.
- Use the same Hydra configs for local and CUDA runs.
- Record CUDA device name, CUDA version, PyTorch version, dtype, seed, and git
  commit in the result manifest.
- Treat CUDA as the throughput path for larger sweeps, but keep CPU checks as
  the numerical reference for small fixtures.
