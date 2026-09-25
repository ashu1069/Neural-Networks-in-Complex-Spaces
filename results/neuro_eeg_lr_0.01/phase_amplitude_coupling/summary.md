# phase_amplitude_coupling

Task: `phase_amplitude_coupling`.

Question: Can models detect phase-amplitude coupling when the high-band amplitude is locked to a low-band phase offset?

Expected signal: full complex/Cartesian/polar views should outperform pure phase or pure magnitude because the label is relational.

Preset: `standard`. Seeds: `[0, 1, 2]`. Examples/class: `128`. Channels: `4`. Time steps: `64`. Train steps: `120`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.3397 | [0.3269, 0.3558] | 4.0713 | 13736 | 27264 | 0.18 |
| `real_stacked` | 0.2949 | [0.2404, 0.3365] | 6.1118 | 13012 | 12960 | 0.17 |
| `real_polar` | 0.9583 | [0.9519, 0.9712] | 0.0930 | 19156 | 19104 | 0.14 |
| `real_phase` | 0.2564 | [0.2308, 0.2692] | 6.5161 | 13012 | 12960 | 0.14 |
| `real_magnitude` | 0.2532 | [0.2115, 0.2981] | 1.3954 | 6868 | 6816 | 0.17 |
