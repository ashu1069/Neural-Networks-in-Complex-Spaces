# Project Plan — Neural Networks in Complex Spaces

This plan turns the thought experiment from the
[motivating blog post](https://blog.gopenai.com/what-happens-if-we-have-complex-valued-neural-networks-a-thought-experiment-ba8dba3784ca)
into a concrete research and engineering roadmap.

The primary goal is **paper-quality empirical evidence**. A reusable exploratory
library is a secondary artifact that should emerge from the experiments, not the
other way around.

The plan is organized into **phases**. Each phase has a goal, a small set of
deliverables, and an exit criterion so we know when to move on.

---

## Guiding research questions

1. **Activation question.** Given the holomorphy/boundedness conflict
   (Liouville), which complex activations give the best trade-off between
   gradient quality, stability, and expressivity in practice?
2. **Parameter-efficiency question.** Does a complex network with `N` complex
   parameters outperform a real network with `2N` real parameters on tasks
   where the input has natural phase structure?
3. **Domain question.** On which tasks (audio, RF, MRI, time-series with
   Fourier features) does the complex inductive bias actually pay off — and
   where is it neutral or harmful?
4. **Optimization question.** Do Wirtinger-calculus-based optimizers
   meaningfully change training dynamics versus naive split-real/imag SGD or
   Adam?

---

## Execution constraints

- Primary development hardware is **Apple Silicon**.
- CUDA servers are available for scale-up and final confirmation runs.
- Treat CPU as the correctness reference. Use MPS only after each operation is
  verified against CPU, because complex tensor coverage and performance can vary
  by PyTorch version and operator.
- Use CUDA for larger experiment sweeps after the CPU smoke tests and local MPS
  checks pass.
- Keep first-pass experiments small enough to run locally. Larger runs should be
  optional scale-up jobs with the exact same configs.
- Every result used in the paper must record environment metadata: Python,
  PyTorch, macOS, device, dtype, seed, git commit, and dataset version.

---

## Phase 0 — Reproducible repo & hardware audit (week 1)

**Goal:** make the repo a working, reproducible Python project and establish
which complex-valued operations are reliable on the local Apple Silicon setup.

- [x] Add `pyproject.toml` with PyTorch, NumPy, SciPy, einops, `hydra-core`,
      pytest, ruff, and mypy.
- [x] Pin the supported Python and PyTorch versions.
- [x] Set up `cvnn/` package skeleton, `tests/`, `experiments/`, `notebooks/`.
- [x] Add CI (GitHub Actions): lint (`ruff`), type-check (`mypy`), `pytest`.
- [x] Pre-commit hooks for formatting.
- [x] Add `scripts/check_torch_complex_support.py` to compare CPU vs MPS/CUDA
      for complex matmul, linear layers, convolutions, FFTs, autograd, and
      selected activations.
- [x] Add `docs/apple_silicon_notes.md` documenting supported ops, fallbacks,
      dtype choices, and expected local runtime.
- [x] Add a result manifest format that records config, seed, metrics, hardware,
      dependency versions, and git commit.

**Exit criterion:** `uv sync --locked --all-groups` works; `uv run pytest`
runs green; CI workflow is committed; CPU/MPS support notes are committed.

---

## Phase 1 — Minimal experimental core (weeks 2–3)

**Goal:** build only the complex-valued components needed to run the first
controlled experiments. Breadth comes later.

Initial components:

- [x] `ComplexLinear`
- [x] Minimal complex MLP builder
- [x] Utilities: `to_complex`, `as_real_pair`, magnitude/phase splitters
- [x] Complex weight initializers: Glorot/He variants in rectangular and polar form
- [x] `ComplexDropout`

Deferred until a benchmark requires them:

- `ComplexConv1d/2d`, `ComplexConvTranspose2d`
- `ComplexBatchNorm` (whitening 2x2 covariance, per Trabelsi et al.)
- `ComplexLayerNorm`

**Tests:**

- [x] Shape and dtype tests for every layer.
- [x] Numerical-gradient checks via `torch.autograd.gradcheck` against
  `torch.complex128` parameters.
- [x] Equivalence test: a `ComplexLinear` with real-only weights should match a
  real `nn.Linear`.
- [x] CPU vs MPS/CUDA agreement tests for supported ops, with explicit skips for
  documented unsupported paths.

**Exit criterion:** the minimal MLP stack is covered by unit tests; `gradcheck`
passes on CPU; supported accelerator paths agree with CPU within tolerance.

**Local status:** CPU and MPS tests pass. CUDA is wired into the tests and audit
script, but still needs to be run on a CUDA server before large sweeps.

---

## Phase 2 — Activations & the Liouville trade-off (weeks 4–5)

**Goal:** implement and *characterize* the main candidate activations so we
can reason about the trade-off empirically.

Implement:

- [x] `CReLU(z) = ReLU(Re z) + i·ReLU(Im z)` — split, not holomorphic.
- [x] `zReLU(z) = z if arg(z) ∈ [0, π/2] else 0` — phase-gated.
- [x] `modReLU(z) = ReLU(|z| + b) · z / |z|` — magnitude-gated, preserves phase.
- [x] `complex_cardioid(z) = 0.5 · (1 + cos(arg z)) · z`.
- [x] `Siglog(z) = z / (1 + |z|)` — bounded, smooth, not holomorphic.
- [x] `complex_tanh(z) = tanh(z)` — meromorphic with poles; included as a cautionary
  baseline.

For each activation, produce a one-page **characterization report**:

- [x] Domain plot (heatmap of `|f(z)|` and `arg f(z)` over a grid in `ℂ`).
- [x] Where is it (non-)holomorphic? Cauchy–Riemann residual map.
- [x] Singularities / blow-ups.
- [x] Edge-case definition at `z = 0`, especially for `arg(z)` and `z / |z|`.
- [x] Empirical gradient norm distribution at init for a fixed MLP.

**Exit criterion:** script-backed characterization report committed for each
activation; results summarized in a single comparison table with fixed metrics
and seeds.

**Local status:** reports are generated by
`uv run python scripts/characterize_activations.py` and committed under
`notebooks/activation_characterization/`.

---

## Phase 3 — Baselines & optimization validity (weeks 6–7)

**Goal:** make the comparisons defensible before running expensive benchmarks.

- [x] Verify PyTorch's complex autograd convention against hand-derived
  Wirtinger-gradient checks for a battery of real-valued losses.
- [x] Use stock `torch.optim.AdamW` / `SGD` as the primary optimizer baseline.
- [x] Add custom optimizer wrappers only if they test a concrete hypothesis not
  covered by stock PyTorch behavior.
- [x] Loss functions: complex MSE, magnitude MSE, phase-aware loss
  (`|y - ŷ|² + λ · 1 - cos(arg y - arg ŷ)`).
- [x] Define baseline families:
  - real network on stacked `(real, imag)` channels;
  - real network with matched scalar parameter count;
  - real network with matched FLOPs where feasible;
  - exact real reparameterization of the complex model for sanity checks.
- [x] Define tuning budget per model family before benchmark runs.

**Exit criterion:** end-to-end training of a tiny complex MLP on a synthetic
task converges and matches a closed-form solution to within tolerance; baseline
rules and tuning budget are documented.

**Local status:** `uv run python experiments/synthetic/complex_linear_regression.py`
converges to the closed-form complex least-squares solution; baseline and
optimizer rules are documented in `docs/baselines_and_optimization.md`.

---

## Phase 4 — Evidence benchmarks (weeks 8–11)

**Goal:** measure CVNNs against real-valued baselines on tasks where phase
matters. Each task gets a CVNN model, a real-valued baseline with **matched
parameter count** (and a separate one with matched FLOPs), and a fixed eval
protocol.

Paper-track tasks:

1. [x] **Synthetic — phase classification / controlled complex regression.** Sanity
   checks with known decision boundaries or closed-form targets.
2. [x] **Synthetic RF modulation classification.** Documented IQ + AWGN
   stand-in with per-SNR snapshot metrics and swept conv results.
3. [ ] **Real RF modulation classification.** RadioML 2018.01A loader/sweep is
   implemented for the local HDF5 archive; paper-quality numbers still need a
   CUDA run with accuracy per SNR bucket.
4. [ ] **Optional scale-up — Audio source separation on STFT.** MUSDB18 or a small
   subset; metric is SI-SDR. Compare CVNN U-Net vs. real-valued U-Net on
   `(real, imag)` stack.
5. [ ] **Optional scale-up — MRI reconstruction (single-coil).** fastMRI knee
   subset; metric is SSIM/PSNR on magnitude images.

For each task:

- Config-driven (Hydra) so runs are reproducible.
- Fixed seeds, ≥3 seeds per condition, report mean ± std.
- Logging via TensorBoard or Weights & Biases.
- Confidence intervals or bootstrap intervals for primary comparisons.
- Store raw per-seed metrics, not just aggregate tables.
- Run CPU smoke tests for correctness and MPS/CUDA runs for throughput when
  supported.

**Exit criterion:** for each chosen task, a table of {CVNN, real-matched-
params, real-matched-FLOPs} with metric, training time, parameter count,
confidence interval, and exact reproduction command.

**Local status:** the synthetic phase-classification benchmark and synthetic RF
modulation benchmark are implemented. Swept results now distinguish the primary
matched shared-trial comparison from independently tuned family winners, so
parameter/FLOP matching is auditable from `summary.md` and `summary.json`.
RadioML ingestion is implemented against the local `GOLD_XYZ_OSC.0001_1024.hdf5`
archive and fixed class sidecars; RadioML paper numbers, audio, and MRI remain
future real-data/scale-up tracks.

---

## Phase 5 — Analysis & paper write-up (weeks 12–14)

**Goal:** consolidate findings into a paper-style technical report before
extracting a general-purpose library API.

- [x] Activation comparison: which one wins where, and does the holomorphy
  defect actually correlate with worse training?
- [x] Parameter-efficiency analysis across tasks.
- [x] Failure-mode catalog (what breaks, what diverges, what overfits).
- [x] Threats to validity: hardware limits, dataset subset bias, tuning budget,
  stochastic variation, and baseline capacity.
- [x] A paper-style technical report (`docs/report.md`) and a follow-up blog post
  that closes the loop on the original thought experiment.

**Exit criterion:** report + blog draft committed; reproducibility instructions
verified by running every experiment from a clean checkout.

---

## Phase 6 — Exploratory library extraction

**Goal:** convert the experimentally validated components into a clean library
without disrupting the paper evidence trail.

- [x] Promote stable experiment code into public `cvnn` APIs.
- Keep paper configs and result manifests immutable.
- [x] Add examples based on the validated synthetic and RF tasks.
- [ ] Write API docs only for components used in successful experiments.

**Exit criterion:** a reader can install the package, reproduce the paper
experiments, and reuse the validated components in a new script.

---

## Stretch goals

- Complex-valued attention and a small complex Transformer for audio.
- Quaternion-valued extension as a comparison point.
- Hardware notes: where `torch.complex64` is and isn't supported on GPU/MPS.

## Risks & open issues

- **PyTorch complex coverage.** Some ops (certain conv variants, AMP) have
  partial complex support. Track gaps in `docs/torch_complex_gaps.md`.
- **Apple Silicon constraints.** MPS can be valuable for local iteration, but
  the paper should not rely on unsupported or silently falling-back operations.
- **Baseline fairness.** Easy to accidentally under-tune the real baseline;
  budget equal hyperparameter-search effort per condition.
- **Dataset access.** fastMRI and MUSDB18 require registration/agreements;
  document this in each experiment's README.

## Definition of done (for the whole project)

A reader who clones the repo can:

1. Install dependencies and run unit tests.
2. Reproduce at least two paper-track benchmark results with a single command
   each.
3. Inspect raw per-seed metrics, configs, environment manifests, and analysis
   notebooks.
4. Read a clear summary of *when* complex-valued networks help, *when* they
   don't, and *which* activation choice survived contact with real data.
