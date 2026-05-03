# Apple Silicon Notes

This project uses Apple Silicon for local development, but CPU remains the
correctness reference for paper-track results.

CUDA servers are available for scale-up and final confirmation runs. Keep local
Mac runs small and use the same configs on CUDA once the CPU smoke test passes.
See `docs/cuda_notes.md`.

## Local Policy

- Run every new complex-valued operation on CPU first.
- Treat MPS as an acceleration path only after CPU/MPS agreement is measured.
- Record Python, PyTorch, macOS, device, dtype, seed, git commit, and dataset
  version in every result manifest.
- Keep first-pass experiments small enough to run locally before scaling.

## Support Audit

Run the audit with `uv`:

```bash
uv run python scripts/check_torch_complex_support.py
```

For a stricter check that exposes unsupported MPS kernels instead of allowing
fallback behavior, run:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python scripts/check_torch_complex_support.py
```

The audit reports gaps by default without failing the process. Add
`--fail-on-gap` when you want a nonzero exit code for unsupported or mismatched
target-device behavior:

```bash
uv run python scripts/check_torch_complex_support.py --fail-on-gap
```

The script compares CPU against the target device for:

- complex matrix multiplication
- complex linear layers
- complex 1D and 2D convolutions
- FFTs
- autograd through a real-valued loss
- the PyTorch issue #119088 complex scalar multiplication reproducer
- simple activation candidates: CReLU, Siglog, and cardioid

When an operation fails on MPS, keep the CPU result as the reference and record
the failure in `docs/torch_complex_gaps.md`.

## Latest Local Audit

As of May 3, 2026 on this Apple Silicon machine:

- Python: 3.12.12
- PyTorch: 2.11.0
- macOS/platform: macOS 14.6 arm64
- MPS available: yes
- Dtype: `torch.complex64`

CPU passed all checked operations. MPS passed complex matmul, 1D convolution,
2D convolution, FFT, autograd through a real-valued loss, the PyTorch issue
#119088 complex scalar multiplication reproducer, CReLU, Siglog, and cardioid.
MPS failed complex linear with:

```text
RuntimeError: mps linear does not support complex types
```

PyTorch issue #119088 was about mixed real/complex scalar binary operations on
MPS, for example multiplying an MPS tensor by `1j`. That upstream issue was
closed by PyTorch PR #119318. The current bottleneck for this project is a
separate operation-level gap: native MPS complex `linear`.

## Dtype Guidance

- Use `torch.complex64` for local MPS throughput checks.
- Use `torch.complex128` on CPU for numerical checks such as `gradcheck`.
- Do not mix dtype changes with model changes in the same experiment.

## Reproducibility

Each experiment should write a manifest using the schema in
`docs/result_manifest.schema.json`. The example in `results/manifest.example.json`
shows the intended shape.
