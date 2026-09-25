# reference_phase_augmented

Task: `phase_locking`.

Question: Does random reference-phase augmentation recover robustness to the unseen fixed reference shift?

Expected signal: phase-aware models should improve relative to the unaugmented reference-shift condition.

Preset: `full`. Seeds: `[0, 1, 2, 3, 4]`. Examples/class: `512`. Channels: `4`. Time steps: `128`. Train steps: `350`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.9995 | [0.9985, 1.0000] | 0.0033 | 54344 | 108288 | 0.62 |
| `real_stacked` | 1.0000 | [1.0000, 1.0000] | 0.0003 | 51748 | 51648 | 0.52 |
| `real_polar` | 0.9995 | [0.9985, 1.0000] | 0.0007 | 76324 | 76224 | 0.66 |
| `real_phase` | 0.9990 | [0.9971, 1.0000] | 0.0121 | 51748 | 51648 | 0.79 |
| `real_magnitude` | 0.2500 | [0.2500, 0.2500] | 1.3867 | 27172 | 27072 | 0.78 |
