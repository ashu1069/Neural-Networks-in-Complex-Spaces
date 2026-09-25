# potential_inverse

Task: `potential_inverse`.

Question: Can models infer which potential generated the observed final wavefunction after 1D Schrodinger evolution?

Expected signal: full complex/Cartesian/polar inputs should outperform phase-only or density-only views when both amplitude and phase carry scattering information.

Preset: `full`. Seeds: `[0, 1, 2, 3, 4]`. Examples/class: `512`. Grid: `128`. Train steps: `400`.

## Plots

![accuracy bar](accuracy_bar.png)

![accuracy by class](accuracy_by_class.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.8863 | [0.8514, 0.9149] | 0.2719 | 11018 | 2704000 | 1.80 |
| `real_stacked` | 0.8894 | [0.8663, 0.9145] | 0.3061 | 5669 | 696480 | 1.34 |
| `real_polar` | 0.9227 | [0.9035, 0.9427] | 0.2196 | 5829 | 716960 | 1.39 |
| `real_phase` | 0.8792 | [0.8412, 0.9086] | 0.2995 | 5669 | 696480 | 1.18 |
| `real_magnitude` | 0.6749 | [0.6380, 0.7094] | 0.7124 | 5509 | 676000 | 0.88 |
