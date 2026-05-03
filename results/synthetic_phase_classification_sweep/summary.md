# Synthetic Phase Classification (Swept)

Selected configuration per model family from a random-search sweep of `16` trials x `3` seeds, following `docs/tuning_budget.md`. See `tuning_log.md` for the per-trial log and `trials.json` for the full record.

Activation (complex): `crelu`. Activation (real baselines): `relu`. `n_classes=8`, `noise_std=0.3`.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 4 | 0.7870 | 0.7656 | 0.0145 | 176 | hidden_features=8, learning_rate=0.005517, steps=400 |
| `real_stacked` | 8 | 0.7854 | 0.7643 | 0.0156 | 184 | hidden_features=16, learning_rate=0.01693, steps=200 |
| `real_matched_params` | 7 | 0.7821 | 0.7630 | 0.0006 | 657 | hidden_features=32, learning_rate=0.03542, steps=200 |
| `real_matched_flops` | 11 | 0.7837 | 0.7650 | 0.0153 | 327 | hidden_features=8, learning_rate=0.007326, steps=200 |
