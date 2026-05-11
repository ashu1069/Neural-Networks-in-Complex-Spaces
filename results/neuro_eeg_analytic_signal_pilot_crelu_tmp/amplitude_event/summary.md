# amplitude_event

Task: `amplitude_event`.

Question: Can models classify which sensor carries an amplitude burst when phase is independent of the label?

Expected signal: magnitude, polar, Cartesian, and complex views should learn; phase-only should remain near chance.

Preset: `standard`. Seeds: `[0, 1, 2]`. Examples/class: `128`. Channels: `4`. Time steps: `64`. Train steps: `120`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.4936 | [0.4327, 0.5769] | 3.0636 | 13736 | 27264 | 0.07 |
| `real_stacked` | 0.3109 | [0.2981, 0.3173] | 5.4739 | 13012 | 12960 | 0.04 |
| `real_polar` | 0.9327 | [0.9135, 0.9423] | 0.1398 | 19156 | 19104 | 0.05 |
| `real_phase` | 0.2564 | [0.2019, 0.2885] | 6.6889 | 13012 | 12960 | 0.04 |
| `real_magnitude` | 0.9455 | [0.8365, 1.0000] | 0.1101 | 6868 | 6816 | 0.04 |
