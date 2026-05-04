# Where the complex inductive bias lives: an architecture-level prior and a separable optimization-level robustness asymmetry, characterized at small scale

**(Concept & Feasibility submission — NeurIPS main track.)**

## Abstract

The intuition that complex-valued neural networks (CVNNs) should help on
phase-bearing signals is widely held and rarely measured carefully. We
demonstrate, at the scale where a mechanism analysis remains tractable,
that the complex inductive bias on IQ classification *decomposes* into
two separable effects with distinct origins. (i) An **architectural**
prior: `ComplexConv1d` is exactly phase-equivariant, and we prove this in
two lines; the magnitude-logit head makes the network phase-invariant at
the readout. (ii) An **optimization-level** asymmetry: under
matched-shared-trial hyperparameter selection, real-valued baselines at
identical capacity exhibit a step-1 explosion-into-dead-region failure
mode on the classifier head that complex networks do not. Per-step
gradient telemetry across 5 activations × 4 model families × 3 seeds = 60
instrumented runs locates the explosion specifically on `head.weight`
(50–80× larger gradient at step 1 for real baselines than for complex)
and shows it produces 3/9 dead seeds (final loss = ln 3, the 3-class
chance level) under three of the five activations, zero dead seeds under
the other two. The same threshold behavior replicates on synthetic
AWGN-only data when the same matched-trial hyperparameters are applied —
the mechanism is hp-driven, not data-driven. We argue that conflating the
two effects produces the inflated apparent CVNN gaps in some prior work,
and that the appropriate framing for CVNN evaluation is to control for
both effects independently. The 1-D scope, three-class subset of
RadioML 2018.01A, and modest hardware are deliberate: the scope is
chosen to support a complete mechanism analysis with end-to-end
reproducibility (manifests, dirty-bit tracking, structural CI guards on
artifacts). Code, sweep harness, telemetry traces, and figure scripts are
released; full-archive scale-up is parameterized via a single CLI flag
and is signposted future work.

## 1. Concept and feasibility scope

### What this paper claims and does not claim

We make three concrete claims, each at a defensible scale and with
explicit reproducibility:

1. **Architectural claim (formal):** `ComplexConv1d` is phase-equivariant
   in the bias-free case (Lemma, §4.1); the magnitude-logit head is
   phase-invariant. A real `Conv1d` on stacked `(I, Q)` channels does not
   get this for free — it must learn rotation invariance from data via a
   2×2 channel mix the complex network gets by construction.
2. **Empirical claim:** complex-valued networks beat capacity-matched
   real-valued ones on synthetic IQ modulation classification (§3.3) and
   on a 3-class subset of RadioML 2018.01A (§3.4) under matched-budget
   evaluation. Conversely, on a task with no temporal phase structure
   (1-D phase classification, §3.2), the complex inductive bias offers
   no measurable advantage. Both directions of evidence are reported.
3. **Mechanism claim:** the apparent magnitude of the win on RadioML is
   activation-dependent (§3.4.1) because matched-shared-trial selection
   picks high-lr regimes under some activations that real baselines
   cannot tolerate. Per-step gradient telemetry (§3.4.4) localizes the
   failure to a step-1 head-gradient explosion that puts a fraction of
   real-baseline seeds into a dead-ReLU region they cannot escape; the
   threshold behavior replicates on synthetic data. Phase equivariance
   of the *layer* is necessary for the architectural prior, but
   activation-level phase equivariance does not predict the empirical
   robustness asymmetry — locating the asymmetry one level lower, in
   `(real, imag)` parameter coupling at the optimization level, is what
   makes the picture self-consistent (§4.1).

We do *not* claim:

- That complex-valued networks beat real-valued ones at full RadioML
  2018.01A scale (24 modulations × 26 SNR levels × 1024-sample length).
  The infrastructure for that experiment is in place
  (`experiments/rf/sweep_radioml.py --preset full`), but the run is
  deferred. The mechanism analysis is the contribution; scale-up is
  signposted future work.
- That the inductive bias survives in 2-D regimes (audio STFT, MRI).
  `ComplexConv2d` is not implemented; the project is 1-D only.
- That CReLU activations generalize across signal-processing settings.
  The activation ablation §3.4.1 is part of the contribution precisely
  because activation choice changes the appearance (though not the
  direction) of CVNN-vs-real gaps.

### Why feasibility-scale is the right scope

A mechanism-level study requires the ability to (a) instrument every
parameter's gradient at every step, (b) run hundreds of training
configurations within a working budget, and (c) confirm cross-task
generalization with a second data distribution. All three are tractable
at the scope chosen — 60 instrumented runs per data source, a
re-runnable budgeted sweep, a synthetic stand-in matching the real-data
benchmark on architecture and hyperparameter regime. They are not
tractable for paper deadlines at 24-class × 1024-sample × 100-seed
scale. We treat the smaller scope as a positive feature: it lets us
report what is happening, not just that something is happening.

### Three benchmarks

The paper reports results on three benchmark tracks, in order of
increasing realism:

- **Synthetic phase classification (§3.2).** A 1-D complex sample whose
  phase encodes one of 8 sectors, with magnitude jitter and AWGN. The
  no-temporal-structure control: the lemma in §4.1 predicts no CVNN
  benefit, and that is what we measure.
- **Synthetic RF modulation classification (§3.3).** Length-128 IQ
  sequences from three PSK constellations (BPSK, QPSK, 8PSK) with
  controlled AWGN at 7 SNR levels. Tests whether sequence-level phase
  rotations exposed to convolution produce a CVNN advantage.
- **RadioML 2018.01A subset (§3.4).** Same scaffold on a gated real-data
  3-class subset of the standard 2018.01A archive (BPSK/QPSK/8PSK at 8
  even SNR levels, sample length 128, 256 examples per (class, SNR)
  bucket). Pulse shaping, carrier-frequency offset, and channel effects
  are present in the data but not specifically modeled.

### Reproducibility as a first-class concern

Every empirical claim above is backed by a committed result manifest
that records the Python / PyTorch / OS / device / dtype, every seed
used, the exact `git_commit` of the producing code, and a `git_dirty`
boolean. CI surfaces newly added dirty manifests so they are reviewed
deliberately rather than slipping through. A structural CI guard checks
the activation characterization artifact regenerates without errors and
contains all six expected activation rows. The companion repository
holds 122 unit and integration tests that gate every push.

The companion repository
([github.com/ashu1069/Neural-Networks-in-Complex-Spaces](https://github.com/ashu1069/Neural-Networks-in-Complex-Spaces))
includes: the `cvnn` library (complex layers, activations, baselines,
losses, initializers), the budgeted random-search harness
(`experiments/_sweep.py`), the gradient telemetry module
(`experiments/rf/gradient_telemetry.py`), all sweep result manifests,
all 120 telemetry JSONL traces (60 RadioML + 60 synthetic), and the
figure-rendering script that produces every figure in this report from
the committed JSON artifacts.

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
  benchmark regen doesn't flip the bit on itself; CI surfaces newly added or
  modified dirty manifests for review.

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

### 3.4 RadioML 2018.01A — the durable finding is robustness, not a 27 pp accuracy gap

[Figures: bars on the original CReLU run
[`radioml_modulation_swept.png`](../results/figures/radioml_modulation_swept.png),
per-SNR breakdown
[`radioml_per_snr.png`](../results/figures/radioml_per_snr.png),
sweep pareto
[`radioml_sweep_pareto.png`](../results/figures/radioml_sweep_pareto.png),
**activation ablation
[`radioml_activation_ablation.png`](../results/figures/radioml_activation_ablation.png)**.]

Same conv architecture, same sweep harness, same matched shared-trial
selection — applied to the gated DeepSig RadioML 2018.01A archive
(BPSK / QPSK / 8PSK at SNRs `[-10, -6, -2, 2, 6, 10, 14, 18]` dB,
`max_per_class_per_snr=256`, sample_length 128) instead of the synthetic
stand-in. After a 16×6 budgeted sweep on an NVIDIA A100, the corrected
`CReLU` run produced a striking 23 pp gap against the best real baseline:

| family | test acc | std | params |
|---|---:|---:|---:|
| **`complex` (CReLU)** | **0.7293** | 0.0085 | 58,886 |
| `real_stacked` | 0.4583 | 0.1394 | 29,891 |
| `real_matched_params` | 0.4999 | 0.1327 | 58,413 |
| `real_matched_flops` | 0.4245 | 0.1415 | 117,123 |

Read in isolation this looks like a dramatic complex-vs-real win at
matched parameters — and that's what an earlier draft of this section
claimed. **The activation ablation forces a different reading.**

#### 3.4.1 Activation ablation reveals the headline gap is mostly
real baselines collapsing

We re-ran the same sweep four more times, varying only the complex side's
activation (the real baselines always use `ReLU`). Headline numbers per
activation, matched-shared-trial selection:

| activation | complex | best real | gap (pp) | complex std | mean real std |
|---|---:|---:|---:|---:|---:|
| `crelu` | 0.7293 | 0.4999 | **+22.94** | 0.009 | 0.138 |
| `cardioid` | 0.7282 | 0.5028 | **+22.54** | 0.019 | 0.134 |
| `siglog` | 0.7014 | 0.4689 | **+23.25** | 0.016 | 0.139 |
| `modrelu` | 0.6683 | 0.6940 | **−2.58** | 0.031 | 0.045 |
| `zrelu` | 0.7330 | 0.7039 | **+2.91** | 0.015 | 0.015 |

The plot ([`radioml_activation_ablation.png`](../results/figures/radioml_activation_ablation.png))
makes the asymmetry visible at a glance: the **CVNN line is nearly flat
across all five activations** (test accuracy 0.668–0.733, range 0.065),
while the real-baseline lines **swing wildly** between ~0.45 and ~0.70
depending on which activation the *complex side* used.

Two readings of this together:

- **The 22–23 pp gap against the best real baseline on `crelu` /
  `cardioid` / `siglog` is mostly the
  real baselines failing under matched-shared-trial selection, not the
  complex network pulling ahead.** When the matched-trial selection
  picks a configuration that complex tolerates (high LR, large hidden
  width — the complex network's per-trial val accuracy is robust to
  these), but real baselines do *not* tolerate (their seed-to-seed std
  jumps to 0.13–0.14, meaning roughly half the seeds barely train), the
  reported gap is dominated by the real side's failure mode.
- **The robustness asymmetry is the durable, defensible finding.**
  Complex tested accuracy varies by 6.5 pp across activations; the
  matched-parameter real baseline varies by 24 pp. Complex's
  seed-to-seed std stays in [0.009, 0.031] across all five runs; the
  real-baseline std varies from 0.015 (`zrelu`) up to 0.139 (`siglog`).
  In the activation regimes where the real baseline is stable
  (`modrelu`, `zrelu`), the complex-vs-real gap drops to ±3 pp.

The honest one-sentence headline:
**complex-valued networks are substantially more robust across activation
choice and seed than capacity-matched real-valued ones on this RadioML
subset; under matched-shared-trial selection the reported accuracy gap
is dominated by that robustness asymmetry rather than by complex
out-classifying real on a level playing field.**

#### 3.4.2 Per-SNR breakdown (CReLU run)

At −10 dB everyone is near chance (~33% for 3 classes; noise wins). From
+2 dB onward, the `CReLU` complex network rises quickly and reaches 92.1% at
+10 dB and 91.6% at +18 dB while the real baselines plateau around 47–61%.
This is the same pattern as the table above — the gap reflects real baselines
failing to train at the selected hyperparameters under `CReLU`, not complex
pulling cleanly away under every stable activation regime. Per-SNR breakdowns
are now captured for all activation runs; the run-level robustness story above
is still the generalizable one.

#### 3.4.3 How this compares to published RadioML numbers

The RadioML 2018.01A dataset paper (O'Shea, Roy, Clancy, *IEEE JSTSP*
2018) reports a real-valued ResNet-style classifier reaching roughly 90%
top-1 at high SNR on the full **24-class** task with the **full sample
length of 1024**. Subsequent CVNN-on-RadioML work (e.g., Krzyston et al.
2020, Tu et al. 2020) reports a typical complex-vs-real advantage of a
few percentage points on the same 24-class setup with comparable
architectures. **Our +3-and-flat ablation result on `modrelu` and `zrelu`
(the activations under which real baselines are also stable) is
consistent with that small reported gap.** The 22–23 pp gap against the best
real baseline on
`crelu` / `cardioid` / `siglog` is *not* a contradiction of the literature
— it's the matched-shared-trial selection rule magnifying real-baseline
instability that the literature's separate-tuning protocols would not
expose.

Our setup uses a deliberately reduced regime — 3-class PSK subset
(BPSK / QPSK / 8PSK), 8 even SNR levels, sample length 128,
`max_per_class_per_snr=256` — so absolute accuracies are not directly
comparable to the literature's 24-class × 26-SNR × 1024-sample
evaluations. Whether the robustness asymmetry survives at that scale is
an open question (see §6).

Reported in
[`results/radioml_modulation_sweep_crelu/summary.md`](../results/radioml_modulation_sweep_crelu/summary.md)
(corrected CReLU run) and
`results/radioml_modulation_sweep_{modrelu,cardioid,siglog,zrelu}/summary.md`
(ablation runs).

#### 3.4.4 Mechanism — explosion-into-dead-region at step 1

[Figures: per-step train loss
[`radioml_telemetry_loss.png`](../results/figures/radioml_telemetry_loss.png),
total gradient norm
[`radioml_telemetry_grad.png`](../results/figures/radioml_telemetry_grad.png).]

§3.4.1 reports an *asymmetry* without explaining it. The `experiments/rf/gradient_telemetry.py`
module re-runs the matched-shared-trial selected configuration with per-step
instrumentation (train loss, total parameter gradient norm, per-parameter
gradient norm, max parameter magnitude). For each of the 5 activations × 4
families × 3 seeds = 60 runs we capped at 200 steps (the divergence pattern
appears within the first 5). The mechanism is unambiguous:

**On the three "unstable" activations** (`crelu`, `cardioid`, `siglog`),
matched-shared-trial picks a high-LR / wide-hidden config that complex
selects because it tolerates it (lr ∈ {0.024, 0.024, 0.040}, hidden = 64).
At step 1, every family sees a large AdamW step away from random init. For
real baselines this produces a step-1 train loss in `[3, 36]`, a total
gradient norm in `[6, 42]`, and a `head.weight` gradient in `[6, 41]`
(i.e. nearly all of the explosion sits on the final classifier's
weights). For the complex network the same step produces a step-1 loss
in `[1.07, 1.28]`, total gradient norm `[0.3, 1.0]`, `head.weight`
gradient `[0.1, 0.7]` — about 50–80× smaller than its real counterparts
on identical data and learning rate.

The complex network's `(real, imag)` parameter coupling distributes the
loss signal across more parameter slots, so the same lr × CE-gradient
product produces a smaller effective step on `head.weight`. The real
network does not have that smoothing; AdamW's first step pushes some
seeds into a region where the second `Conv1d`'s ReLU activations are
all-zero (or all-active and saturated), gradients vanish, and training
"stabilizes" at uniform-prediction chance level. We confirmed this:
**the dead seeds end at exactly `loss = ln(3) ≈ 1.099`** (3-class CE
under uniform predictions), with `total_grad_norm ≤ 0.05`, and zero
test accuracy improvement from random init.

Dead-real-seed counts (out of 9 per activation, three families × three
seeds):

| activation | dead seeds | step-1 head.weight grad (real, max) |
|---|:--:|---:|
| `crelu` (lr=0.024) | 3 / 9 | 13.7 |
| `cardioid` (lr=0.024) | 3 / 9 | 19.5 |
| `siglog` (lr=0.040) | 3 / 9 | 40.4 |
| `modrelu` (lr=0.008) | **0 / 9** | 0.17 |
| `zrelu` (lr=0.0024) | **0 / 9** | 0.13 |

**The dead-seed rate jumps from 0% to 33% as soon as the matched-shared-trial
selected lr crosses ~0.02.** Two orders of magnitude separation in the
step-1 head gradient between the unstable and stable activation regimes.
Under `modrelu` and `zrelu` no real-baseline seed dies, complex's step-1
gradient drops to `[0.05, 0.10]`, and the test-accuracy gap collapses to
±3 pp.

**Implication for the paper's headline.** The "complex is more robust to
activation choice" finding is the empirical surface of a sharper
mechanistic claim: *the complex parameterization is more tolerant of
high effective per-step updates on the classifier head.* The matched-
shared-trial rule selects configurations that exploit this tolerance;
naive selection rules (or independent-tuning protocols) hide it. A
practitioner choosing between `ComplexConv1d + |z|² head` and a
real-conv-stack baseline at fixed parameter budget should expect the
complex variant to be substantially less LR-fragile at the head; this is
the property to reach for, not "complex always wins by 27 pp."

**Cross-task confirmation: the mechanism is hp-driven, not data-driven.**
[Figures: `synthetic_rf_telemetry_loss.png` /
`synthetic_rf_telemetry_grad.png`.] To check whether the
explosion-into-dead-region pattern is RadioML-specific (pulse shaping,
carrier offset, channel effects) or generic to the hp regime, we re-ran
the same 60-run telemetry against the synthetic AWGN-only RF generator
([`experiments.rf.synthetic_modulation`](../experiments/rf/synthetic_modulation.py))
using the same per-activation hyperparameters that the RadioML
matched-shared-trial selected. Side-by-side dead-seed counts:

| activation | RadioML dead/9 | synthetic dead/9 | RadioML max real grad | synthetic max real grad |
|---|:--:|:--:|---:|---:|
| `crelu` | 3 | 3 | 19.9 | 34.1 |
| `cardioid` | 3 | 3 | 19.9 | 34.1 |
| `siglog` | 3 | 2 | 42.1 | 55.7 |
| `modrelu` | **0** | **0** | 0.3 | 0.5 |
| `zrelu` | **0** | **0** | 0.2 | 0.7 |

The pattern replicates almost exactly: same dead-seed counts on
crelu/cardioid (3/3), one less on siglog (2/3 vs 3/3), and zero deaths
on modrelu/zrelu in both. Step-1 gradients on synthetic are *larger*
than RadioML's (AWGN-only signals carry more per-sample variance than
pulse-shaped, channel-attenuated ones) but the threshold behavior — the
0%→33% jump at lr ~0.02 — is the same. **The matched-shared-trial
selection rule, not the data distribution, is what triggers the
explosion.** Stored under
[`results/synthetic_rf_telemetry/`](../results/synthetic_rf_telemetry/).

**Bug surfaced.** The first attempt at the RadioML sweep used the
synthetic benchmark's odd-stepped SNR list, which the loader silently
dropped because RadioML 2018.01A only ships even SNRs (2 dB steps from
−20). The fix landed in
[`experiments/rf/radioml.py`](../experiments/rf/radioml.py): the loader
now raises by default when any requested `(modulation, SNR)` bucket is
empty, with `strict_snr=False` to opt back into silent skipping for
exploratory work.

## 4. Interpretation

The three findings together support a more nuanced falsifiable story
than the original "complex wins" framing:

> **The complex inductive bias pays for itself when the task carries
> structure that complex multiplication naturally encodes — and is
> neutral otherwise. On real-data IQ classification, complex networks are
> *robust to activation choice and seed* in a way capacity-matched real
> networks are not; this robustness asymmetry, rather than a raw accuracy
> advantage on a level playing field, is what produces large reported
> gaps under matched-shared-trial selection.**

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
3. **Activation choice changes magnitudes, not direction — and exposes
   the methodological asymmetry of matched-shared-trial selection.**
   §3.4.1's ablation shows that swapping activations leaves complex's
   accuracy nearly flat (range 0.065) while real baselines swing by roughly
   0.24 for the matched-parameter baseline and 0.29 across all real baselines.
   The pattern (complex wins or ties on RF, ties on phase) holds across
   all five activations; the *size* of the win on RadioML depends on
   whether the selected configuration happens to be one that real
   baselines can train at. The robustness-across-activations asymmetry
   is the durable finding; the apparently large per-activation gap is a
   consequence of that asymmetry, not a separate phenomenon.

### 4.1 Why complex helps on IQ data — phase equivariance

The "complex multiplication has the right semantics for phase-bearing
signals" intuition can be made formal in one short lemma.

**Lemma (phase equivariance of `ComplexConv1d`).** *Let*
`C(z)[t] = Σ_τ K[τ] · z[t-τ] + b` *be a `ComplexConv1d` layer with
complex-valued kernel `K`, complex bias `b`, and complex input sequence
`z`. For any global phase `φ ∈ ℝ` and any `t`,*

> `C(e^{iφ} · z)[t] = e^{iφ} · C(z)[t] + (1 - e^{iφ}) · b`.

*Proof.* By linearity of complex multiplication and convolution,
`C(e^{iφ}·z)[t] = Σ_τ K[τ] · (e^{iφ}·z[t-τ]) + b = e^{iφ} · Σ_τ K[τ] · z[t-τ] + b
= e^{iφ}·(C(z)[t] - b) + b`. ∎

In the bias-free case (or after subtracting the bias) the result is
exactly equivariant. The biased case adds a fixed offset `(1 - e^{iφ})·b`
that the next layer can absorb; in our networks the head's `|·|` magnitude
operation is itself phase-invariant
(`|e^{iφ} · y| = |y|` for real `φ`), so the *output* logits are unaffected
by a constant input phase shift — the architecture is phase-invariant
as a whole at the readout.

The same does not hold for `Conv1d` operating on a stacked
`(Re z, Im z)` real input. A global phase shift `e^{iφ}` corresponds to
multiplying each `(I, Q)` sample by the 2×2 rotation matrix
`R_φ = [[cos φ, −sin φ], [sin φ, cos φ]]`. To match `ComplexConv1d`'s
equivariance, every real conv kernel would need to commute with `R_φ`
applied to its 2 input channels — which forces a constraint on the
kernel that real `Conv1d` does not impose by construction. The real
network *can* learn the symmetry from data, but it has to spend capacity
on it; the complex network gets it for free.

**This is the structural inductive bias the synthetic phase
classification result (§3.2) and the RadioML result (§3.4) probe in
opposite regimes.** A single complex sample (phase classification) gives
no temporal structure for convolutional equivariance to bite on, and the
2-D real network has the same "see real and imag" feature view, so the
complex network buys nothing — the §3.2 null is exactly what the lemma
predicts. A length-`L` IQ sequence (RadioML) does have temporal
structure, the carrier phase varies sample-to-sample under channel
effects, and a phase-equivariant conv stack matches that symmetry — the
§3.3 / §3.4 wins are exactly what the lemma predicts.

#### What the lemma does *not* explain

Phase equivariance of the *layer* is necessary for the architectural
prior, but the activations between conv layers vary in their own
equivariance. By direct computation:

- `modrelu(z) = ReLU(|z| + b) · z/|z|` is phase-equivariant
  (`|·|` is invariant; `z/|z|` is equivariant).
- `siglog(z) = z / (1 + |z|)` is phase-equivariant for the same reason.
- `crelu(z) = ReLU(Re z) + i ReLU(Im z)` is *not* — splitting
  real and imaginary coordinates breaks rotation invariance.
- `cardioid(z) = ½(1 + cos(arg z)) z` is *not* — `arg(e^{iφ}z) = arg z + φ`,
  so the cosine factor depends on `φ`.
- `zrelu(z) = z` if `arg z ∈ [0, π/2]` else `0` is *not* — the quadrant
  gate depends on the basis.

If the §3.4 robustness asymmetry were *driven* by activation phase
equivariance, the stable activations should be exactly `modrelu` and
`siglog`. They are not: §3.4.4 shows `modrelu` and `zrelu` are stable
(0/9 dead seeds across 9 real-baseline runs each), while
`crelu`, `cardioid`, and `siglog` are unstable (3/9 dead). One
phase-equivariant activation is on each side of the divide. **Phase
equivariance of the layer is necessary for the inductive bias, but the
robustness asymmetry §3.4.4 measures lives one level lower — at the
optimization dynamics induced by `(real, imag)` parameter coupling on
the classifier head, not at the equivariance of the activation.** The
two effects are partially independent and partially compounding. This
also explains why the §3.4 result is most defensible at activations like
`zrelu` where complex wins by a small margin rather than at `crelu`
where the apparent gap is amplified by the optimization-side mechanism.

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
- **One loss.** Classification runs use cross-entropy. We swept complex
  activations on the RadioML subset, but did not try magnitude-MSE /
  phase-aware losses on these classification tasks.
- **No `ComplexBatchNorm` or `ComplexLayerNorm`.** Deferred from Phase 1
  per the original plan; landing them would let us train deeper conv
  stacks fairly.
- **CPU and Apple Silicon for development; one A100 for the headline RF
  sweep.** No multi-machine reproducibility check.

## 6. Future work

- **Full RadioML clean-room rerun.** The activation ablation is comparable
  across the same eight even SNR levels; a full clean-room rerun before
  submission would still be useful for independent confirmation.
- **Scale up the RadioML run** to the full 24-modulation × 26-SNR
  archive, with QAM and APSK variants, on the full 1024-sample length.
  Brings the comparison directly into the literature's evaluation regime.
  *The infrastructure for this is in place* —
  `experiments/rf/sweep_radioml.py --preset full` switches the
  modulations / SNR levels / sample length / search space to the
  full-archive bundle (see [`docs/radioml.md`](radioml.md#sweep-presets)
  for the recommended GPU command and cost estimates). The run was
  deferred from the headline write-up to keep the paper anchored on the
  3-class subset that supports the mechanism analysis (§3.4.4); we will
  execute the full sweep on review feedback or before any venue
  submission, whichever comes first. Recommended first activation is
  `zrelu` (the only ablation point where real baselines also train
  cleanly, so the comparison is least confounded by the
  matched-shared-trial selection's interaction with §3.4.4's explosion
  mechanism).
- `ComplexBatchNorm` (whitening 2x2 covariance, per Trabelsi et al.) and
  deeper conv stacks.
- Full per-SNR uncertainty bands for the RadioML activation runs, not just
  mean per-SNR curves.
- Audio (MUSDB18 STFT) and single-coil MRI reconstruction (fastMRI) — the
  two tasks the original plan named as scale-up targets.
- A separate study of per-activation training dynamics in the RF setting:
  does ModReLU's bias-sign regime visibly affect convergence?

## 7. Reproducibility

A C&F submission's value is in *what others can build on*. Every claim
above is backed by a single CLI invocation against committed artifacts:

```bash
# 1. Reproduce all three benchmarks (sweeps + selections + manifests)
uv sync --all-groups
uv run python experiments/synthetic/sweep_phase_classification.py
uv run python experiments/rf/sweep_synthetic_modulation.py --device cuda
uv run python experiments/rf/sweep_radioml.py --device cuda \
    --activation crelu --seeds 0 1 2 3 4 5
# (repeat for cardioid / siglog / modrelu / zrelu for the §3.4.1 ablation)

# 2. Reproduce the mechanism analysis (§3.4.4 / §4.1)
uv run python scripts/run_gradient_telemetry.py
uv run python scripts/run_synthetic_rf_telemetry.py

# 3. Regenerate every figure in this report from committed JSON
uv run python scripts/generate_paper_figures.py

# 4. Optional: scale up to the full RadioML archive
uv run python experiments/rf/sweep_radioml.py --device cuda \
    --preset full --activation zrelu --seeds 0 1 2
```

Each result manifest records its `git_commit`, environment, and a
`git_dirty` flag computed on a tree that excludes `results/` and the
activation-characterization output directory (so regen does not flip the
bit on itself). CI surfaces newly added or modified manifests with
`git_dirty: true` as warnings on the PR. A structural CI guard checks
the committed activation characterization artifact contains all six
expected activation rows after a regen.

Headline run provenance:
- §3.2 phase classification sweep: macOS/CPU at git `f7d995c`.
- §3.3 synthetic RF sweep: NVIDIA A100-SXM4-40GB.
- §3.4 RadioML sweep + activation ablation: NVIDIA A100-SXM4-40GB across
  five clean-tree runs (one per activation).
- §3.4.4 RadioML telemetry + §3.4.4-cross synthetic telemetry: macOS/CPU
  at git `c024f72` and `b716f9c` respectively.
- §4.1 phase-equivariance lemma: derivable in two lines, no compute.

The 122-test suite (`uv run pytest`) covers every module the empirical
claims depend on, including a self-contained test that exercises the
gradient-telemetry pipeline against a synthetic RadioML-shape HDF5
fixture in tmp_path so external data is not required for CI.
