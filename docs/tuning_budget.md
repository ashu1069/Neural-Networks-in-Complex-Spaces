# Tuning Budget

A benchmark is only as honest as the hyperparameter effort spent on each
condition. This document fixes the rules so Phase 4 comparisons can't be
quietly tilted by under-tuning the baselines.

## The rule

**Every model family in a benchmark gets the same hyperparameter search
budget.** No exceptions. If the complex model gets 32 trials, every real
baseline gets 32 trials, drawn from the same search distribution.

"Family" means each entry from `docs/baselines.md`: complex, naive
real-stacked, matched-parameter real, matched-FLOP real, exact reparam (the
exact reparam is a witness, not tuned).

## The default budget

For each Phase 4 benchmark, unless an experiment-specific README overrides:

- **Trials per family:** 16 hyperparameter samples drawn from a fixed
  search distribution per task.
- **Seeds per trial:** 3 random seeds.
- **Selection:** pick the best trial by mean validation metric across the 3
  seeds. Report mean +/- std on the held-out test set for that trial.
- **Search space:** a single space defined per task (in
  `experiments/<task>/README.md`), shared across all families. Specifying
  per-family search spaces is allowed only for hyperparameters that don't
  exist in one of the families (e.g. `modrelu_init_bias` only applies to
  complex models that use `ModReLU`); even then, document the asymmetry.

## What counts against the budget

- Every distinct `(hyperparameter sample, seed)` training run consumes 1 of
  `trials * seeds` slots per family.
- Restarts due to bugs do not count, but only if the bug is documented in
  the experiment's README and the run was not used for selection.
- Pilot runs to fix the search range do not count, but the final search
  range must be frozen before the budgeted runs begin.

## What does *not* count

- Running the exact-reparameterization sanity check.
- Re-running the *selected* trial across additional seeds to widen error
  bars on the final reported number (these run after selection and don't
  influence which model wins).

## How to record it

Each benchmark's results directory must contain a `tuning_log.md` (or
equivalent) with one entry per trial: family, hyperparameter sample, seed,
final metric, wall time, git commit. Selection must be deterministic from
this log.

## Why this matters

If the complex model's search space happens to bracket the right learning
rate and the real baseline's doesn't, the complex model "wins" for the
wrong reason. Phase 3's job is to make that failure mode mechanical to spot:
the budget rule above is the mechanical part.
