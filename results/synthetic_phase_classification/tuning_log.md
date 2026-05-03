# Tuning Log - Synthetic Phase Classification

This run is a **snapshot, not a swept benchmark**. It does not satisfy the
16 trials x 3 seeds rule from `docs/tuning_budget.md`. It exists to verify
the harness ends-to-end and to give a reference table while the synthetic
task's difficulty is being tuned.

## What was run

| Family | Trials | Seeds/Trial | Total runs | Selection |
| --- | ---: | ---: | ---: | --- |
| `complex` | 1 | 3 | 3 | none |
| `real_stacked` | 1 | 3 | 3 | none |
| `real_matched_params` | 1 | 3 | 3 | none |
| `real_matched_flops` | 1 | 3 | 3 | none |

All families used identical hyperparameters (see `summary.json` `config`
block). No search; no selection. Numbers in `summary.md` are descriptive,
not selected.

## Status

- This snapshot will be **replaced** by a budgeted sweep before the result
  is cited in any paper-track context.
- Once the synthetic task is recalibrated to a regime where families
  differ, perform 16 trials x 3 seeds for each family with a fixed
  per-task search distribution and rewrite this log.
