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
| `complex` | 0.8800 | [0.8467, 0.9094] | 0.2967 | 11018 | 2704000 | 2.42 |
| `real_stacked` | 0.8431 | [0.8267, 0.8600] | 0.3917 | 5669 | 696480 | 0.98 |
| `real_polar` | 0.8800 | [0.8627, 0.8969] | 0.3139 | 5829 | 716960 | 0.74 |
| `real_phase` | 0.8322 | [0.8020, 0.8624] | 0.3699 | 5669 | 696480 | 0.97 |
| `real_magnitude` | 0.6745 | [0.6373, 0.7118] | 0.7136 | 5509 | 676000 | 0.63 |
