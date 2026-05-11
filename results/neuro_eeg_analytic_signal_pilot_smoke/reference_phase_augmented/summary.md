# reference_phase_augmented

Task: `phase_locking`.

Question: Does random reference-phase augmentation recover robustness to the unseen fixed reference shift?

Expected signal: phase-aware models should improve relative to the unaugmented reference-shift condition.

Preset: `smoke`. Seeds: `[0]`. Examples/class: `24`. Channels: `4`. Time steps: `32`. Train steps: `25`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.3500 | [0.3500, 0.3500] | 1.5762 | 2280 | 4480 | 0.01 |
| `real_stacked` | 0.4000 | [0.4000, 0.4000] | 1.2911 | 2164 | 2144 | 0.01 |
| `real_polar` | 0.4000 | [0.4000, 0.4000] | 1.3859 | 3188 | 3168 | 0.01 |
| `real_phase` | 0.5000 | [0.5000, 0.5000] | 1.1257 | 2164 | 2144 | 0.01 |
| `real_magnitude` | 0.2500 | [0.2500, 0.2500] | 1.3853 | 1140 | 1120 | 0.01 |
