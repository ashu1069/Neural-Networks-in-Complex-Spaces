# phase_amplitude_coupling

Task: `phase_amplitude_coupling`.

Question: Can models detect phase-amplitude coupling when the high-band amplitude is locked to a low-band phase offset?

Expected signal: full complex/Cartesian/polar views should outperform pure phase or pure magnitude because the label is relational.

Preset: `standard`. Seeds: `[0, 1, 2]`. Examples/class: `128`. Channels: `4`. Time steps: `64`. Train steps: `120`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.2853 | [0.2404, 0.3365] | 5.1178 | 13736 | 27264 | 0.08 |
| `real_stacked` | 0.2308 | [0.1923, 0.2788] | 6.4411 | 13012 | 12960 | 0.04 |
| `real_polar` | 0.3622 | [0.2500, 0.4519] | 2.5008 | 19156 | 19104 | 0.04 |
| `real_phase` | 0.2276 | [0.1827, 0.2596] | 6.1207 | 13012 | 12960 | 0.04 |
| `real_magnitude` | 0.2436 | [0.2308, 0.2500] | 1.3882 | 6868 | 6816 | 0.04 |
