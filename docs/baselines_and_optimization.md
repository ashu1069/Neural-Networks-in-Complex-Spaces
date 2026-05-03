# Baselines And Optimization (Overview)

This file used to define baselines, losses, optimizer policy, and tuning rules
in one place. Those topics are now owned by focused documents to keep them
from drifting against each other:

- **Baseline families and matching rules:** [`baselines.md`](baselines.md).
- **Hyperparameter tuning policy and budget:** [`tuning_budget.md`](tuning_budget.md).
- **Apple Silicon / MPS notes:** [`apple_silicon_notes.md`](apple_silicon_notes.md).
- **CUDA notes:** [`cuda_notes.md`](cuda_notes.md).
- **PyTorch complex-tensor support gaps:** [`torch_complex_gaps.md`](torch_complex_gaps.md).

## Quick reminders

- **Optimizer:** stock `torch.optim.AdamW` / `SGD`. Do not add a custom
  Wirtinger optimizer wrapper unless there is a concrete hypothesis stock
  PyTorch cannot test.
- **Autograd convention:** for a real scalar `L(x, y)` and `z = x + iy`,
  PyTorch returns `dL/dx + i dL/dy`. Validated by
  `cvnn.optim.validate_pytorch_complex_autograd()`.
- **Losses:** `complex_mse_loss`, `magnitude_mse_loss`, `phase_aware_loss`
  (see [`baselines.md`](baselines.md) for when each is appropriate;
  `phase_aware_loss` has a documented gradient hazard near `z = 0`).
- **Synthetic convergence smoke test:**
  `uv run python experiments/synthetic/complex_linear_regression.py`.

If you are adding a new baseline family, document it in `baselines.md` and
update `tuning_budget.md` if the budget needs to change. Do **not** add a
parallel definition to this file.
