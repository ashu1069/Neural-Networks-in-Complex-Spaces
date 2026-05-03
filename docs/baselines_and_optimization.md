# Baselines And Optimization

Phase 3 exists to make later benchmark comparisons defensible.

## Complex Autograd Convention

For a real scalar objective `L(x, y)` and `z = x + i y`, PyTorch stores the
complex gradient as the real-coordinate steepest-descent direction:

```text
dL/dx + i dL/dy
```

The helper `validate_pytorch_complex_autograd()` checks this convention against
hand-derived gradients for:

- `sum(|z|^2)`
- `sum(Re(z)^2)`
- `sum(Im(z)^2)`
- real-valued complex projections
- complex MSE

Run:

```bash
uv run python - <<'PY'
from cvnn.optim import validate_pytorch_complex_autograd

for check in validate_pytorch_complex_autograd():
    print(check)
PY
```

## Optimizer Policy

Use stock PyTorch optimizers first:

- `torch.optim.AdamW` with `weight_decay=0.0` for initial convergence checks
- `torch.optim.SGD` for simple dynamics comparisons

Do not add custom Wirtinger optimizer wrappers until there is a concrete
hypothesis that stock PyTorch optimizers cannot test.

## Losses

Phase 3 adds:

- `complex_mse_loss(input, target) = |input - target|^2`
- `magnitude_mse_loss(input, target) = (|input| - |target|)^2`
- `phase_aware_loss(input, target) = |input - target|^2 + lambda * (1 - cos(arg target - arg input))`

PyTorch defines `torch.angle(0)` as zero, so zero-valued phases are
deterministic but conventional. Experiments where zero phase matters should
document that choice.

## Baseline Families

Every paper-track task should define these baselines before running sweeps:

- **Complex model:** native complex tensors and complex-valued layers.
- **Stacked real/imag model:** real model receiving real and imaginary parts as
  separate channels.
- **Matched scalar-parameter real model:** real model with approximately the
  same number of real scalar parameters as the complex model.
- **Matched FLOPs real model:** real model sized to match compute where the
  operation count is meaningful.
- **Exact real reparameterization:** block-real representation of the complex
  model, used as a sanity check for implementation equivalence.

## Tuning Budget

Before a benchmark run, define:

- seeds per condition;
- optimizer family and learning-rate grid;
- activation choices;
- initialization choices;
- early-stopping or fixed-step policy;
- maximum wall-clock budget per condition;
- hardware target: CPU smoke, MPS local check, CUDA scale-up.

Use the same search effort for complex and real baselines. If one baseline gets
extra manual tuning, document it as a threat to validity.

## Synthetic Convergence Check

Run the Phase 3 smoke experiment:

```bash
uv run python experiments/synthetic/complex_linear_regression.py
```

This trains a one-layer complex model with stock `AdamW` and compares it against
the closed-form complex least-squares solution.
