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
| `complex` | 0.8510 | [0.8106, 0.8816] | 0.3983 | 11018 | 2704000 | 2.63 |
| `real_stacked` | 0.8384 | [0.8196, 0.8600] | 0.4014 | 5669 | 696480 | 1.20 |
| `real_polar` | 0.8796 | [0.8616, 0.8969] | 0.3136 | 5829 | 716960 | 1.35 |
| `real_phase` | 0.8322 | [0.8020, 0.8624] | 0.3699 | 5669 | 696480 | 1.28 |
| `real_magnitude` | 0.6749 | [0.6380, 0.7106] | 0.7134 | 5509 | 676000 | 1.10 |
