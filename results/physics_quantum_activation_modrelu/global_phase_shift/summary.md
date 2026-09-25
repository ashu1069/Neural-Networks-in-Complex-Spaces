# global_phase_shift

Task: `potential_inverse`.

Question: Do models trained in one global-phase convention respect the physical invariance psi -> exp(i theta) psi?

Expected signal: coordinate-dependent models may degrade under an unseen global phase; magnitude-only is invariant but information-poor.

Preset: `full`. Seeds: `[0, 1, 2, 3, 4]`. Examples/class: `512`. Grid: `128`. Train steps: `400`.

## Plots

![accuracy bar](accuracy_bar.png)

![accuracy by class](accuracy_by_class.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.7898 | [0.7671, 0.8106] | 0.5092 | 11020 | 2704000 | 5.13 |
| `real_stacked` | 0.8404 | [0.8231, 0.8596] | 0.4014 | 5669 | 696480 | 0.85 |
| `real_polar` | 0.8796 | [0.8620, 0.8961] | 0.3141 | 5829 | 716960 | 0.91 |
| `real_phase` | 0.8322 | [0.8020, 0.8624] | 0.3700 | 5669 | 696480 | 1.23 |
| `real_magnitude` | 0.6769 | [0.6408, 0.7125] | 0.7114 | 5509 | 676000 | 1.73 |
