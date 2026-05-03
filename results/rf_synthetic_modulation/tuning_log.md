# Tuning Log - Synthetic RF Modulation Classification

This run is a **snapshot, not a swept benchmark**. It does not satisfy the
16 trials x 3 seeds rule from `docs/tuning_budget.md`. It exists to verify
the harness ends-to-end on a sequence-shaped task and to give an honest
"flatten-MLP-no-temporal-structure" baseline against which a future
`ComplexConv1d` variant or a real RadioML loader can be compared.

## What was run

| Family | Trials | Seeds/Trial | Total runs | Selection |
| --- | ---: | ---: | ---: | --- |
| `complex` | 1 | 3 | 3 | none |
| `real_stacked` | 1 | 3 | 3 | none |
| `real_matched_params` | 1 | 3 | 3 | none |
| `real_matched_flops` | 1 | 3 | 3 | none |

All families used identical hyperparameters (see `summary.json` `config`
block). No search; no selection. Numbers in `summary.md` are descriptive,
not selected. Initial trial of `--modulations bpsk qpsk 8psk qam16 qam64`
showed all families stuck near chance because flatten-then-MLP cannot
recognize 2D constellation patterns from i.i.d. symbols; restricted to
PSK-only (angle-classifiable) modulations for a more meaningful baseline.

## Status

- This snapshot will be **replaced** before the result is cited in any
  paper-track context, by either:
  - 16 trials x 3 seeds with a fixed per-task search distribution, or
  - a `ComplexConv1d`-based architecture that can exploit temporal
    structure (which is the standard approach in modulation-classification
    literature), or
  - swap-in of the real RadioML 2018.01A loader (the directory layout is
    designed so `experiments/rf/radioml.py` can land alongside this file
    and reuse the four-family scaffolding).
