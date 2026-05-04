# Neural Networks in Complex Spaces

A research workspace for building, training, and analyzing **Complex-Valued
Neural Networks (CVNNs)** — networks whose weights, biases, and activations
live in `ℂ` rather than `ℝ`.

This project is a practical follow-up to the thought experiment in
["What happens if we have complex-valued neural networks?"](https://blog.gopenai.com/what-happens-if-we-have-complex-valued-neural-networks-a-thought-experiment-ba8dba3784ca).
The blog posed the question; this repo is the attempt to answer it with code,
experiments, and measurements.

## Why complex numbers?

A complex weight `w = a + ib` simultaneously **scales magnitude** and
**rotates phase**. That makes `ℂ` a natural number system for problems where
signals carry both an amplitude and a phase:

- Wireless / RF and radar signal processing
- Audio and speech (STFT representations are complex)
- MRI reconstruction (k-space data is complex)
- Optics, holography, and SAR imaging
- Any domain currently forced to split a complex signal into two real channels

## The central tension

CVNNs sit on top of an unresolved mathematical conflict:

- **Holomorphy** (complex differentiability via the Cauchy–Riemann equations)
  gives clean, well-behaved gradients.
- **Boundedness** keeps activations stable during training.
- **Liouville's theorem** says no non-constant function can be *both*
  holomorphic *and* bounded on all of `ℂ`.

Real-valued favorites break in the complex plane: `tanh(z)` inherits the
singularities of the trigonometric tangent and blows up periodically. Common
workarounds like **CReLU** (ReLU applied to real and imaginary parts
separately) are not holomorphic and are best understood as engineering
compromises rather than principled solutions.

This repo treats that tension as the **central research question**, not a
footnote.

## Headline findings

Three budgeted benchmarks, four model families each, sweeps following
[`docs/tuning_budget.md`](docs/tuning_budget.md). Full write-up in
[`docs/report.md`](docs/report.md); blog draft in
[`docs/blog_post_draft.md`](docs/blog_post_draft.md).

- **Activation characterization** ([`activation_tradeoff.png`](results/figures/activation_tradeoff.png)) —
  the Liouville bind is empirically visible: every candidate activation
  trades holomorphy for boundedness or vice versa.
- **Synthetic phase classification — null** ([`synthetic_phase_classification_swept.png`](results/figures/synthetic_phase_classification_swept.png)) —
  all four families converge to ~76% with overlapping CIs after 16×3
  sweep. The complex inductive bias offers no measurable advantage when
  the task carries no structure for `ℂ`-multiplication to exploit.
- **Synthetic RF modulation — complex wins, with the smallest model**
  ([`rf_synthetic_modulation_swept.png`](results/figures/rf_synthetic_modulation_swept.png),
  [`rf_sweep_pareto.png`](results/figures/rf_sweep_pareto.png)) — 16×6 GPU
  sweep with a `ComplexConv1d` stack: complex reaches 0.819 test accuracy
  at 3,974 parameters; the next-best real baseline reaches 0.781 at 7,779
  parameters. CIs do not overlap, Welch t-test gives `p < 0.01`.
- **RadioML 2018.01A — complex wins by 21–28 pp on real data**
  ([`radioml_modulation_swept.png`](results/figures/radioml_modulation_swept.png),
  [`radioml_per_snr.png`](results/figures/radioml_per_snr.png)) — same
  scaffold on the gated DeepSig archive (BPSK / QPSK / 8PSK subset, 16×6
  on A100): complex 0.722 vs best real 0.511. Per-SNR breakdown: at
  ≥+10 dB the complex network reaches ~92% while real baselines plateau
  at 50–60%. Real baselines also have ~6× higher seed-to-seed std — half
  the seeds barely train. Complex is more accurate *and* more reliable.

The headline thesis: **the complex inductive bias pays for itself when the
task carries structure that complex multiplication naturally encodes, and
is neutral otherwise. On real-data IQ classification with channel effects,
the gap grows substantially.**

## Scope and caveats

Everything is **1-D only** — `ComplexLinear` and `ComplexConv1d`,
benchmarks on scalar inputs and length-128 IQ sequences. No
`ComplexConv2d`, no images, no audio STFT, no MRI yet. The RadioML run
uses a 3-modulation × 8-SNR subset of the full archive at sample length
128; the full 24-class × 26-SNR × 1024-sample setup is opt-in but not yet
swept. See [`docs/report.md`](docs/report.md#5-limitations) for the full
limitations list.

## Goals

1. Provide clean PyTorch implementations of complex-valued layers, activations,
   initializations, and optimizers.
2. Benchmark CVNNs against equivalent real-valued baselines on tasks where
   phase information matters (audio, RF, MRI-style reconstruction).
3. Empirically compare candidate activations (modReLU, zReLU, CReLU, complex
   cardioid, Siglog, etc.) on stability, expressivity, and gradient quality.
4. Document what works, what doesn't, and *why*, with reproducible code.

## Status

Paper-track synthetic benchmarking and write-up artifacts are in place. The
repo now has a `uv`-managed
Python project, CI configuration, pre-commit hooks, a minimal `cvnn` package,
complex tensor utilities, complex initializers, `ComplexLinear`,
`ComplexConv1d`, `ComplexDropout`, a minimal complex MLP builder, Phase 2
complex activations, script-backed activation characterization reports, Phase 3
losses/autograd checks, synthetic phase-classification and RF-modulation
benchmarks with matched real baselines, sweep/result artifacts, a report/blog
draft, result manifest helpers, and MPS/CUDA complex-tensor support audit
tooling. Implementation milestones are tracked in
[PROJECT_PLAN.md](PROJECT_PLAN.md).

## Planned repository layout

```text
.
├── cvnn/                  # core package and reproducibility helpers
│   ├── analysis/          # script-backed characterization utilities
│   ├── activations/       # Phase 2 activation implementations
│   ├── init/              # Phase 1 complex-aware initializers
│   ├── layers/            # Phase 1 complex-valued layers
│   ├── optim/             # Phase 3 optimizer experiments
│   ├── nn.py              # minimal complex MLP builder
│   ├── utils.py           # complex tensor conversion/splitting helpers
│   └── repro.py           # result manifest helpers
├── docs/                  # hardware notes, schemas, support-gap tracking
├── experiments/           # task-specific training scripts + configs
├── notebooks/             # exploratory analyses and visualizations
├── results/               # lightweight committed result manifests
├── scripts/               # local audit and utility scripts
├── tests/
├── pyproject.toml
├── uv.lock
├── PROJECT_PLAN.md
└── README.md
```

## Getting started

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy cvnn scripts experiments
```

Generate Phase 2 activation reports with:

```bash
uv run python scripts/characterize_activations.py
```

The comparison table is written to
[notebooks/activation_characterization/comparison.md](notebooks/activation_characterization/comparison.md).

Run the Phase 3 synthetic optimization check with:

```bash
uv run python experiments/synthetic/complex_linear_regression.py
```

Run the Phase 4 synthetic phase-classification benchmark with:

```bash
uv run python experiments/synthetic/phase_classification.py
```

The benchmark writes raw per-seed rows, aggregate summaries, and a manifest to
`results/synthetic_phase_classification/` by default. Use `--device cuda` on a
CUDA server after the CPU smoke run is green.

Run the synthetic RF modulation benchmark with:

```bash
uv run python experiments/rf/synthetic_modulation.py
```

Regenerate the paper figures from committed artifacts with:

```bash
uv run python scripts/generate_paper_figures.py
```

On Apple Silicon, audit local complex-tensor support with:

```bash
uv run python scripts/check_torch_complex_support.py
```

For a stricter MPS check, disable fallback behavior:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python scripts/check_torch_complex_support.py
```

Add `--fail-on-gap` if you want unsupported MPS behavior to produce a nonzero
exit code.

On a CUDA server, run the same audit against CUDA:

```bash
uv run python scripts/check_torch_complex_support.py --device cuda
```

## Background reading

- Trabelsi et al., *Deep Complex Networks* (ICLR 2018)
- Trabelsi et al. reference code:
  [ChihebTrabelsi/deep_complex_networks](https://github.com/ChihebTrabelsi/deep_complex_networks)
- Hirose, *Complex-Valued Neural Networks* (Springer)
- Arjovsky et al., *Unitary Evolution Recurrent Neural Networks*
- Virtue et al., *Complex-Valued CNNs for MRI Fingerprinting*
- The motivating blog post linked above

## License

Apache License 2.0 — see [LICENSE](LICENSE).
