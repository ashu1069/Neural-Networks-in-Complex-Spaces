# phase_locking

Task: `phase_locking`.

Question: Can models classify inter-channel phase locking when amplitude envelopes are randomized independently of the label?

Expected signal: phase-aware and Cartesian views should learn; magnitude-only should remain near chance.

Preset: `standard`. Seeds: `[0, 1, 2]`. Examples/class: `128`. Channels: `4`. Time steps: `64`. Train steps: `120`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.9263 | [0.8269, 1.0000] | 0.3894 | 13736 | 27264 | 0.07 |
| `real_stacked` | 0.9808 | [0.9519, 1.0000] | 0.0296 | 13012 | 12960 | 0.04 |
| `real_polar` | 0.9712 | [0.9423, 0.9904] | 0.0747 | 19156 | 19104 | 0.04 |
| `real_phase` | 1.0000 | [1.0000, 1.0000] | 0.0050 | 13012 | 12960 | 0.04 |
| `real_magnitude` | 0.2500 | [0.2500, 0.2500] | 1.3877 | 6868 | 6816 | 0.04 |
