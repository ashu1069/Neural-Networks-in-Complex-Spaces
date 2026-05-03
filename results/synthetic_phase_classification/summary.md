# Synthetic Phase Classification

Snapshot of one configuration. Numbers depend on platform BLAS, wall-clock load, and `git_commit`/`git_dirty` recorded in the manifest. Re-runs will not be byte-identical; see `docs/baselines.md` and `docs/tuning_budget.md` for the comparison rules.

Activation (complex): `crelu`. Activation (real baselines): `relu`. Seeds: `[0, 1, 2]`. Steps: `400`. `n_classes=8`, `noise_std=0.3`.

| model | hidden | params | est. forward MAdds | accuracy mean | accuracy std | 95% CI | loss mean | train s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 16 | 336 | 576 | 0.7617 | 0.0052 | [0.7578, 0.7676] | 0.606 | 0.19 |
| `real_stacked` | 16 | 184 | 160 | 0.7565 | 0.0050 | [0.7510, 0.7607] | 0.588 | 0.083 |
| `real_matched_params` | 30 | 338 | 300 | 0.7588 | 0.0026 | [0.7568, 0.7617] | 0.602 | 0.089 |
| `real_matched_flops` | 58 | 646 | 580 | 0.7601 | 0.0025 | [0.7578, 0.7627] | 0.624 | 0.12 |
