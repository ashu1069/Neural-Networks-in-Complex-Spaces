# phase_amplitude_coupling

Task: `phase_amplitude_coupling`.

Question: Can models detect phase-amplitude coupling when the high-band amplitude is locked to a low-band phase offset?

Expected signal: full complex/Cartesian/polar views should outperform pure phase or pure magnitude because the label is relational.

Preset: `standard`. Seeds: `[0, 1, 2]`. Examples/class: `128`. Channels: `4`. Time steps: `64`. Train steps: `120`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.2821 | [0.2500, 0.3077] | 2.9834 | 13736 | 27264 | 0.19 |
| `real_stacked` | 0.2949 | [0.2885, 0.2981] | 3.1140 | 13012 | 12960 | 0.13 |
| `real_polar` | 0.9038 | [0.8077, 0.9615] | 0.2733 | 19156 | 19104 | 0.11 |
| `real_phase` | 0.2564 | [0.1827, 0.3462] | 3.3262 | 13012 | 12960 | 0.12 |
| `real_magnitude` | 0.2436 | [0.1923, 0.2692] | 1.4091 | 6868 | 6816 | 0.12 |
