# PyTorch Complex Support Gaps

Use this file to track operations that fail, fall back unexpectedly, or show
large CPU/MPS numerical disagreement.

## Current Status

Last local audit: May 3, 2026

- Machine/platform: macOS 14.6 arm64
- Python: 3.12.12
- PyTorch: 2.11.0
- Command: `uv run python scripts/check_torch_complex_support.py`
- Dtype: `torch.complex64`
- MPS available: yes

| Operation | CPU | MPS | Max MPS abs error | Notes |
| --- | --- | --- | ---: | --- |
| matmul | pass | pass | 6.743e-07 |  |
| linear | pass | error |  | `RuntimeError: mps linear does not support complex types` |
| conv1d | pass | pass | 3.016e-06 |  |
| conv2d | pass | pass | 3.016e-06 |  |
| fft | pass | pass | 9.830e-07 |  |
| autograd | pass | pass | 2.403e-07 | Real-valued `abs() ** 2` loss |
| issue_119088_complex_scalar_mul | pass | pass | 0.000e+00 | Regression check for PyTorch issue #119088 |
| activation_crelu | pass | pass | 0.000e+00 |  |
| activation_siglog | pass | pass | 5.961e-08 |  |
| activation_cardioid | pass | pass | 1.201e-07 |  |

Phase 1 implements `ComplexLinear` with explicit complex matmul rather than
native `torch.nn.functional.linear`. Local CPU/MPS agreement tests pass through
that path, so the native MPS complex-linear gap is documented but no longer
blocks minimal local model prototyping.

PyTorch issue #119088 was fixed upstream via PR #119318 for mixed real/complex
scalar binary operations on MPS. It explains why `x.to("mps") * 1j` is now a
useful regression check, but it does not remove the need for our complex-linear
fallback.

CUDA audit: pending. Run
`uv run python scripts/check_torch_complex_support.py --device cuda` on the CUDA
server before trusting large benchmark runs.

## Audit Template

```text
Date:
Machine:
macOS:
Python:
PyTorch:
Command:
PYTORCH_ENABLE_MPS_FALLBACK:

Operation:
Device:
Dtype:
Status:
Observed behavior:
Fallback or workaround:
```
