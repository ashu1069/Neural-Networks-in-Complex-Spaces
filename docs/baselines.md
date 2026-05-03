# Real-Valued Baselines

When a Phase 4 benchmark claims a complex-valued network outperforms a
real-valued one, the comparison is only meaningful if both models had a fair
budget. This document defines the four baseline families used in this project
and the rules for matching them against a complex reference model.

## The four families

For every Phase 4 benchmark, we run the complex model **plus** these four
real-valued baselines:

### 1. Naive real-stacked

A real network that treats `(real, imag)` as two stacked feature channels.
Implemented as `cvnn.baselines.RealStackedLinear` (and analogous helpers for
deeper architectures).

- **Layout:** input shape `(..., 2 * in_features)` with the first
  `in_features` entries holding real parts and the next `in_features` holding
  imaginary parts. Output uses the same convention.
- **Parameter count:** for `in_features = F` and `out_features = F'`,
  `2F * 2F' + 2F' = 4 F F' + 2 F'`. This is roughly **twice** the real
  parameter count of an equivalent `ComplexLinear`
  (`2(F F' + F') = 2 F F' + 2 F'`).
- **Why we run it:** it's what most "I tried doing this with a regular
  network" baselines look like. Useful as the upper-bound real baseline.

### 2. Matched-parameter real

A real network whose total real parameter count is the closest feasible match
to the complex model's, counted via `cvnn.baselines.count_real_parameters`
(which doubles every complex parameter's `numel`). Exact equality is possible
for some layer widths but not guaranteed for discrete hidden-channel choices.

- **How to match:** start from the naive-real shape and choose hidden width(s)
  that minimize
  `abs(count_real_parameters(real_model) - count_real_parameters(complex_model))`.
  Always report the actual counts in the result table so the residual mismatch
  is visible.
- **Why we run it:** controls for "the complex model only wins because it
  effectively has more parameters."

### 3. Matched-FLOP real

A real network whose forward-pass FLOP count equals the complex model's.
Complex multiply-add costs 4 real multiplies + 2 real adds; real multiply-add
costs 1 multiply + 1 add. So a complex `ComplexLinear(F, F')` costs roughly
`F * F' * 6` real FLOPs on the multiply-add path, vs `F * F' * 2` for a real
`Linear(F, F')`.

- **How to match:** size the real network so its FLOP count equals the
  complex model's. For a single linear layer, the matched real layer has
  feature widths scaled by `sqrt(3)` (since `2 * F_real * F'_real ~= 6 * F * F'`
  when `F_real = F * sqrt(3)`).
- **Why we run it:** controls for "the complex model only wins because it
  burns more compute."
- **When to skip:** if a benchmark is dominated by data movement or non-linear
  ops (e.g. attention), FLOP matching adds noise rather than signal. Document
  the skip in the experiment's README.

### 4. Exact real reparameterization (sanity check)

The same complex model rewritten in real coordinates with no behavior change.
Built via `cvnn.baselines.complex_to_real_reparam`, which expands a complex
weight `W = A + iB` into the block matrix `[[A, -B], [B, A]]`.

- **Why we run it:** confirms that what we're calling "complex inductive
  bias" really is a bias (a constraint on the function class), not just a
  reparameterization. If the matched-parameter real baseline reaches the
  same loss as the exact reparameterization, the inductive bias gave us
  *nothing* on this task.
- **Use:** assert that `complex_model(z)` and
  `real_reparam(stack_real_imag(z))` agree to floating-point tolerance.
  Then *don't train the reparameterization*; it's a conformance witness, not
  a baseline to beat.

## Reporting

Every Phase 4 benchmark table must include columns for at least:

- the complex model
- the matched-parameter real baseline
- the matched-FLOP real baseline (or an explicit "N/A - FLOP-matching not
  meaningful here" note)

The naive-real-stacked and exact-reparam columns are optional but encouraged.

Report `count_real_parameters` and total forward FLOPs in every row, so the
matching is verifiable from the table alone.

For swept experiments, distinguish two tables:

- **Matched shared-trial comparison:** choose the reference complex trial by
  validation accuracy, then report every real baseline at that same trial
  index. This is the primary paper comparison because the matched baselines are
  tied to the selected complex model.
- **Independent family winners:** choose each family's best validation trial
  independently. This is useful as a diagnostic, but those rows are not
  necessarily matched to the selected complex model.

## What to record

For every baseline run, the result manifest must capture:

- model family (`complex`, `real_stacked`, `real_matched_params`,
  `real_matched_flops`, `real_reparam`)
- `count_real_parameters` value
- forward FLOPs (one number per forward pass at the eval input shape)
- everything else the result manifest already requires (seed, git commit,
  device, dtype, dataset version)

This makes after-the-fact "wait, were those parameter-matched?" audits cheap.
