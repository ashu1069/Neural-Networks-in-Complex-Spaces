# phase_amplitude_coupling

Task: `phase_amplitude_coupling`.

Question: Can models detect phase-amplitude coupling when the high-band amplitude is locked to a low-band phase offset?

Expected signal: full complex/Cartesian/polar views should outperform pure phase or pure magnitude because the label is relational.

Preset: `standard`. Seeds: `[0, 1, 2]`. Examples/class: `128`. Channels: `4`. Time steps: `64`. Train steps: `120`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.2788 | [0.2308, 0.3558] | 1.9136 | 13736 | 27264 | 0.17 |
| `real_stacked` | 0.2692 | [0.2115, 0.3269] | 1.8077 | 13012 | 12960 | 0.11 |
| `real_polar` | 0.7276 | [0.6731, 0.7596] | 0.7371 | 19156 | 19104 | 0.11 |
| `real_phase` | 0.2244 | [0.1538, 0.2596] | 1.8854 | 13012 | 12960 | 0.12 |
| `real_magnitude` | 0.2372 | [0.2212, 0.2596] | 1.4078 | 6868 | 6816 | 0.11 |
