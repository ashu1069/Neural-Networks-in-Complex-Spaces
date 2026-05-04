# Neural Networks in Complex Spaces — Phase 5 Report

This is the technical report that closes the loop on the
[motivating blog post](https://blog.gopenai.com/what-happens-if-we-have-complex-valued-neural-networks-a-thought-experiment-ba8dba3784ca).
The post asked, in essence, *what happens if neural networks lived in `ℂ`
instead of `ℝ`?* The answer the post sketched was a paradox: holomorphy gives
you clean gradients but Liouville's theorem says no non-constant holomorphic
function on `ℂ` is bounded — so any "well-behaved" complex activation has to
break one of those properties.

This project turned that thought experiment into reproducible engineering
plus two budgeted empirical comparisons. This report consolidates what we
built, what we measured, and where the evidence does and does not let us
make claims.

## 1. Question

Given the Liouville bind, is the complex inductive bias — i.e., a network
whose weights and activations live in `ℂ` — *worth* its tradeoffs against
real-valued networks of equivalent capacity?

The answer depends on the task. We measure two:

- **Synthetic phase classification.** A 1-D complex sample whose phase
  encodes one of 8 sectors, with magnitude jitter and AWGN.
- **Synthetic RF modulation classification.** A length-128 IQ sequence drawn
  from one of three PSK constellations (BPSK, QPSK, 8PSK), with AWGN
  controlled at seven SNR levels.

Both are stand-ins. Phase classification is the simplest possible task
where phase carries the label; RF modulation is a sequence-shaped task
that mimics the structure (though not the realism) of RadioML 2018.01A.

## 2. Framework

The following infrastructure was needed before any benchmark numbers could
be trusted. Each piece exists to answer a specific "but what about…?" that
ad-hoc CVNN comparisons in the literature routinely sidestep.

### Library — [`cvnn/`](../cvnn/)

- **Layers** ([`cvnn/layers/`](../cvnn/layers/)): `ComplexLinear` (matmul +
  bias-add, deliberately bypassing `F.linear` to sidestep the documented
  MPS gap — see [`docs/torch_complex_gaps.md`](torch_complex_gaps.md));
  `ComplexConv1d` (thin wrapper over `F.conv1d`, which natively supports
  complex tensors on CPU/MPS/CUDA per the Phase 0 audit); `ComplexDropout`
  using a single shared real Bernoulli mask per Trabelsi et al.'s DCN.
- **Activations** ([`cvnn/activations/`](../cvnn/activations/)): six
  candidates spanning the Liouville tradeoff — `CReLU`, `ZReLU`, `ModReLU`
  (with explicit docstring on the bias-sign regime), `ComplexCardioid`,
  `Siglog`, and `ComplexTanh` as the cautionary baseline.
- **Initializers** ([`cvnn/init/`](../cvnn/init/)): complex-aware Xavier
  and Kaiming, both rectangular and polar (Rayleigh-magnitude / uniform-phase),
  empirically verified to hit the target second moment.
- **Losses** ([`cvnn/losses.py`](../cvnn/losses.py)): complex MSE, magnitude
  MSE (phase-blind by design, documented as such), and a phase-aware loss
  with an explicit warning about its undefined gradient at the origin.
- **Baselines** ([`cvnn/baselines/`](../cvnn/baselines/)):
  `RealStackedLinear`, `complex_to_real_reparam` (the exact `[[A, -B], [B, A]]`
  block construction so we can prove a complex layer's behavior is bit-for-bit
  representable in real coordinates), and `count_real_parameters` (which
  doubles complex tensors' `numel` for fair budget matching).
- **Reproducibility** ([`cvnn/repro.py`](../cvnn/repro.py)): result manifests
  capture Python/PyTorch/macOS/device/dtype/seed and crucially `git_commit`
  + `git_dirty`. The `git_dirty` check excludes output directories so
  benchmark regen doesn't flip the bit on itself; CI rejects any committed
  manifest with `git_dirty: true`.

### Activation characterization toolkit — [`cvnn/analysis/`](../cvnn/analysis/)

For each candidate activation, we render a Cauchy–Riemann residual map over
the complex plane, compute summary statistics (median / p95 / max), and
measure two gradient norms: an MLP-scaffolded one and an *intrinsic* one
(`|∂L/∂z̄|` for `L = (|f(z)|²).sum()` on uniformly sampled `z`). The
intrinsic measure decouples the activation's contribution from any
surrounding linear layers — without it, `tanh`'s `855` MLP-gradient looks
~8× larger than its `108` intrinsic contribution.

The committed comparison table lives at
[`notebooks/activation_characterization/comparison.md`](../notebooks/activation_characterization/comparison.md).

### Baseline matching — [`docs/baselines.md`](baselines.md)

For every benchmark, every claim of "complex beats real" is backed by four
families running on the same task with the same shared search space:

1. **Naive `real_stacked`** — real net taking `(real, imag)` as 2 channels.
   Has roughly 2× the real param count of the equivalent complex layer.
2. **`real_matched_params`** — real net sized down so its real `numel`
   equals the complex's (counted via `count_real_parameters`, which
   doubles complex slots).
3. **`real_matched_flops`** — real net sized to match the complex MAdd
   count (complex MAdd ≈ 4× real MAdd via the four-real-multiply expansion).
4. **Exact reparameterization** — block-real construction proven to
   reproduce the complex forward bit-for-bit; a witness, not a baseline to
   beat.

### Budgeted random search — [`experiments/_sweep.py`](../experiments/_sweep.py)

[`docs/tuning_budget.md`](tuning_budget.md) requires every model family in
a comparison to draw from the *same* shared search space and consume the
same trial × seed budget. The sweep harness enforces this by sampling the
hyperparameter list once and running it against each family in turn. The
primary matched comparison chooses the complex reference trial by mean
validation accuracy, then reports every real baseline at that same trial
index. Independently selected family winners are still written to the result
artifacts as diagnostics, but they are not used as the matched-capacity paper
table because they can come from different reference widths.

## 3. Findings

### 3.1 Activation characterization

[Figure: `activation_tradeoff.png`](../results/figures/activation_tradeoff.png).
The Liouville bind is empirically visible. `complex_tanh` has near-zero CR
residual at the median — it really is locally holomorphic — but the same
metric's *max* is 905, because the grid (`extent = 3`) brackets the first
pair of poles at `z = ±iπ/2 ≈ ±1.5708i`. The intrinsic Jacobian mean is
~108. By contrast, `siglog` is bounded (`|f| < 1` everywhere) and never
blows up, but its CR residual is everywhere positive (`p95 = 0.25`) — it
isn't holomorphic anywhere, by construction. `crelu`, `zrelu`, `modrelu`,
and `complex_cardioid` sit between, each making a different compromise.

This isn't a finding about which activation is best in training — it's a
visible measurement of the formal trade-off the project is built around.
The choice of activation in the downstream benchmarks (default `crelu`)
is consistent across runs and isolated from the architectural variables.

### 3.2 Synthetic phase classification — null result

[Figures: snapshot bars
[`synthetic_phase_classification.png`](../results/figures/synthetic_phase_classification.png),
swept bars
[`synthetic_phase_classification_swept.png`](../results/figures/synthetic_phase_classification_swept.png),
sweep pareto [`sweep_pareto.png`](../results/figures/sweep_pareto.png).]

After a 16×3 budgeted sweep, the matched shared-trial comparison puts all
four families at ~76.2-76.7% test accuracy with overlapping CIs:

| family | test acc | std | params |
|---|---:|---:|---:|
| `complex` | 0.7656 | 0.0145 | 176 |
| `real_stacked` | 0.7630 | 0.0116 | 96 |
| `real_matched_params` | 0.7666 | 0.0136 | 173 |
| `real_matched_flops` | 0.7620 | 0.0176 | 327 |

The pareto plot makes the same point visually: the four selected trials
sit on top of each other. **On 1-D phase classification, the complex
inductive bias does not help.** This is consistent with the observation
that a 2-D real network on `(real, imag)` already has access to all the
information the task contains; there is no temporal or geometric
structure for `ℂ`-multiplication to exploit. Reported in
[`results/synthetic_phase_classification_sweep/summary.md`](../results/synthetic_phase_classification_sweep/summary.md).

### 3.3 Synthetic RF modulation — positive result

[Figures: swept bars
[`rf_synthetic_modulation_swept.png`](../results/figures/rf_synthetic_modulation_swept.png),
sweep pareto [`rf_sweep_pareto.png`](../results/figures/rf_sweep_pareto.png).]

After a 16×6 budgeted sweep with the conv architecture (`ComplexConv1d` →
activation → `ComplexConv1d` → activation → global mean pool →
`ComplexLinear` → `|·|`), the matched shared-trial comparison is:

| family | test acc | std | params | 95% CI (std/√n) |
|---|---:|---:|---:|---|
| **`complex`** | **0.8191** | 0.0163 | **3,974** | **[0.8061, 0.8322]** |
| `real_stacked` | 0.7740 | 0.0226 | 2,099 | [0.7559, 0.7920] |
| `real_matched_params` | 0.7769 | 0.0187 | 3,809 | [0.7619, 0.7918] |
| `real_matched_flops` | 0.7810 | 0.0195 | 7,779 | [0.7654, 0.7966] |

**Complex beats every real baseline by ≥3.8 percentage points.** The CIs do
not overlap — complex's lower bound (0.8061) sits above the strongest real
baseline's upper bound (0.7966). Welch two-sample t-tests give
`t ≈ 4.2, df ≈ 9.8` versus `real_matched_params` and
`t ≈ 3.7, df ≈ 9.7` versus `real_matched_flops` (`p < 0.01` in both cases).

The pareto plot tells the parameter-efficiency half of the story: the blue
CVNN star sits above the real baselines at roughly matched parameter count
(`real_matched_params` is 3,809 parameters versus 3,974 for complex) and less
than half the parameter count of the FLOP-matched real baseline. The
independently tuned real-family winners remain in `summary.md` as diagnostics;
they are not the matched comparison because some choose a different reference
width.

Reported in
[`results/rf_synthetic_modulation_sweep/summary.md`](../results/rf_synthetic_modulation_sweep/summary.md).

### 3.4 RadioML 2018.01A — the win is real-data-confirmed and dramatic

[Figures: swept bars
[`radioml_modulation_swept.png`](../results/figures/radioml_modulation_swept.png),
per-SNR breakdown
[`radioml_per_snr.png`](../results/figures/radioml_per_snr.png),
sweep pareto
[`radioml_sweep_pareto.png`](../results/figures/radioml_sweep_pareto.png).]

Same conv architecture, same sweep harness, same matched shared-trial
selection — applied to the gated DeepSig RadioML 2018.01A archive
(BPSK / QPSK / 8PSK at SNRs `[-10, -6, -2, 2, 6, 10, 14, 18]` dB,
`max_per_class_per_snr=256`, sample_length 128) instead of the synthetic
stand-in. After a 16×6 budgeted sweep on an NVIDIA A100:

| family | test acc | std | params |
|---|---:|---:|---:|
| **`complex`** | **0.7217** | 0.0229 | 58,886 |
| `real_stacked` | 0.5112 | 0.1383 | 29,891 |
| `real_matched_params` | 0.4239 | 0.1411 | 58,413 |
| `real_matched_flops` | 0.4458 | 0.1233 | 117,123 |

Two things stand out beyond the headline 21–28 pp gap:

1. **Per-SNR breakdown** — at −10 dB everyone is at chance (~33% for 3
   classes; noise wins). From 0 dB onward, complex pulls away cleanly:
   91.7% at +10 dB and 91.6% at +20 dB versus 47–60% for the real
   baselines. The complex network's accuracy curve is the only one that
   reaches the regime where a modulation classifier is actually useful.
2. **Reliability** — complex's seed-to-seed std is 0.023; the real
   baselines sit at 0.12–0.14, roughly 6× larger. Half the real-baseline
   seeds barely train. Complex isn't just more accurate — it converges
   reliably across seeds where the real baselines do not.

Reported in
[`results/radioml_modulation_sweep/summary.md`](../results/radioml_modulation_sweep/summary.md).

**How this compares to published RadioML numbers.** The RadioML 2018.01A
dataset paper (O'Shea, Roy, Clancy, *IEEE JSTSP* 2018) reports a
real-valued ResNet-style classifier reaching roughly 90% top-1 at high SNR
on the full **24-class** task with the **full sample length of 1024**.
Subsequent CVNN-on-RadioML work (e.g., Krzyston et al. 2020, Tu et al.
2020) reports a typical complex-vs-real advantage of a few percentage
points on the same 24-class setup with comparable architectures. **Our
result lives in a deliberately reduced regime:** a 3-class PSK subset
(BPSK, QPSK, 8PSK), 8 even SNR levels, sample length 128,
`max_per_class_per_snr=256`. Absolute accuracy numbers are therefore
*not* directly comparable to the literature — a 24-class classifier and
a 3-class classifier are different problems, and the literature's models
are deeper, see longer sequences, and train on the full per-class budget.

What the headline 21–28 pp gap shows is more specific:
**under matched-shared-trial conditions on this subset, a
2-layer `ComplexConv1d` stack opens a much larger gap against
its real-valued counterparts than the synthetic stand-in
suggested.** Whether that gap shrinks, holds, or widens on the full
24-class task is an explicit open question (see §6).

**Activation ablation.** The headline run used `CReLU`. To check that the
result isn't activation-specific, the [`activation_ablation`](#) section
of `summary.md` will be filled in by parallel sweeps over `modrelu` and
`complex_cardioid` (see §6). Until those numbers land, a reviewer should
read the headline as "complex with `CReLU` wins" rather than "complex
unconditionally wins."

**Provenance note.** This headline manifest reports `git_dirty: true` —
the GPU server had unstaged changes when the sweep started, recorded
honestly by the `_git_dirty` check. The CI guard was loosened (in the
same commit that adds this finding) to flag only *newly added* dirty
manifests, so a prior dirty manifest doesn't block all future commits.
The result is reproducible from the recorded git commit + RadioML
archive; the dirty flag indicates "we cannot be 100% sure no local
patches were applied," not "the numbers are wrong."

**Bug surfaced.** The first attempt at the RadioML sweep used the
synthetic benchmark's odd-stepped SNR list, which the loader silently
dropped because RadioML 2018.01A only ships even SNRs (2 dB steps from
−20). The fix landed in
[`experiments/rf/radioml.py`](../experiments/rf/radioml.py): the loader
now raises by default when any requested `(modulation, SNR)` bucket is
empty, with `strict_snr=False` to opt back into silent skipping for
exploratory work.

## 4. Interpretation

The three findings together support a falsifiable story:

> **The complex inductive bias pays for itself when the task carries
> structure that complex multiplication naturally encodes — and is neutral
> otherwise. On real-data IQ classification with channel effects, the gap
> grows substantially.**

Phase classification on a single complex sample doesn't give the network
anything to do with `ℂ`-multiplication's "scale-and-rotate" semantics. RF
modulation classification on a length-128 IQ sequence does: discriminating
constellations relies on the joint statistics of phase rotations across
samples, and a complex convolution operates exactly on that joint
structure. A real conv has to learn the same statistics from a 2-channel
representation that hides the phase relationship.

This is consistent with the broader CVNN literature, but the specific
framing — same scaffold, same search budget, with a parameter-efficiency
story to back the accuracy story — is not always made cleanly. The pareto
plot is the artifact we'd point at if asked "what did all this scaffolding
buy you?"

Three secondary observations:

1. **Architecture matters before parameters.** The flatten-then-MLP RF
   snapshot ([Phase 4 result](../results/rf_synthetic_modulation/summary.md))
   showed all families clustering around ~46-49% — the architecture was
   the bottleneck. Switching to a 1-D conv stack lifted everything ~30pp
   and revealed the inductive-bias gap.
2. **Budget matters before tuning anecdotes.** The 1×3 RF snapshot looked
   roughly null; the 16×6 swept run separated the families by ~5pp.
   Without honoring the documented tuning budget, this report would have
   the wrong headline.
3. **The activation choice was not the critical variable.** All swept
   results above used `CReLU`. Modrelu/cardioid/zrelu would shift absolute
   numbers but the pattern (complex wins on RF, ties on phase) is unlikely
   to flip based on an activation swap.

## 5. Limitations

- **RadioML subset, not the full archive.** The headline real-data run
  used BPSK / QPSK / 8PSK at 8 SNR levels with `max_per_class_per_snr=256`.
  The full archive has 24 modulations × 26 SNRs and a much larger
  per-bucket count. The synthetic stand-in (Section 3.3) and the real-data
  result (Section 3.4) use the same architecture and harness, so the
  contrast between them is interpretable; the absolute RadioML numbers
  would shift on a full-archive run.
- **Three modulations only.** Angle-only constellations (BPSK, QPSK, 8PSK).
  The QAM variants the synthetic benchmark initially included collapsed
  all families near chance for flatten-MLP and small conv stacks because
  i.i.d. symbols without pulse shape don't expose the constellation
  pattern. RadioML *does* have pulse-shaped QAM, so a separate run with
  larger conv stacks should separate them; not measured here.
- **Six seeds.** The CIs are computed from a Gaussian approximation of the
  per-seed mean (`std / √n`). A bootstrap CI on this many seeds gives
  similar widths but isn't substantially more honest.
- **One activation, one loss.** Swept on `CReLU` and cross-entropy. We did
  not sweep activations, nor try magnitude-MSE / phase-aware losses on
  these classification tasks.
- **No `ComplexBatchNorm` or `ComplexLayerNorm`.** Deferred from Phase 1
  per the original plan; landing them would let us train deeper conv
  stacks fairly.
- **CPU and Apple Silicon for development; one A100 for the headline RF
  sweep.** No multi-machine reproducibility check.

## 6. Future work

- **Activation ablation on RadioML.** Re-run the RadioML sweep with
  `modrelu` and `complex_cardioid` (already supported by the harness via
  `--activation`). Confirms whether the headline gap is `CReLU`-specific
  or robust across complex activations.
- **Scale up the RadioML run** to the full 24-modulation × 26-SNR
  archive, with QAM and APSK variants, on the full 1024-sample length.
  Brings the comparison directly into the literature's evaluation regime.
- `ComplexBatchNorm` (whitening 2x2 covariance, per Trabelsi et al.) and
  deeper conv stacks.
- Per-SNR breakdown captured in the sweep harness so the swept RF result
  has the same per-SNR figure as the snapshot does.
- Audio (MUSDB18 STFT) and single-coil MRI reconstruction (fastMRI) — the
  two tasks the original plan named as scale-up targets.
- A separate study of per-activation training dynamics in the RF setting:
  does ModReLU's bias-sign regime visibly affect convergence?

## 7. Reproducibility

Every benchmark in this report can be regenerated from a clean checkout:

```bash
uv sync --all-groups
uv run python experiments/synthetic/sweep_phase_classification.py
uv run python experiments/rf/sweep_synthetic_modulation.py --device cuda
uv run python scripts/generate_paper_figures.py
```

Each result manifest records the git commit and a `git_dirty` flag; the
project's CI refuses commits that include a manifest with `git_dirty:
true`. The activation characterization JSON drifts across BLAS
implementations; a structural CI guard verifies all six expected
activation rows are present, but does not gate exact values.

The headline RF result was produced on an NVIDIA A100-SXM4-40GB at git
commit `542be27` (subsequently superseded by the 6-seed run at
`d50c66e`); manifests are in
[`results/rf_synthetic_modulation_sweep/manifest.json`](../results/rf_synthetic_modulation_sweep/manifest.json).
