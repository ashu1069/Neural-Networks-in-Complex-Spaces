# amplitude_event

Task: `amplitude_event`.

Question: Can models classify which sensor carries an amplitude burst when phase is independent of the label?

Expected signal: magnitude, polar, Cartesian, and complex views should learn; phase-only should remain near chance.

Preset: `standard`. Seeds: `[0, 1, 2]`. Examples/class: `128`. Channels: `4`. Time steps: `64`. Train steps: `120`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.3205 | [0.2981, 0.3654] | 1.6954 | 13736 | 27264 | 0.25 |
| `real_stacked` | 0.3173 | [0.2308, 0.3942] | 1.6661 | 13012 | 12960 | 0.14 |
| `real_polar` | 0.9519 | [0.9423, 0.9615] | 0.2075 | 19156 | 19104 | 0.12 |
| `real_phase` | 0.2981 | [0.2500, 0.3365] | 1.7881 | 13012 | 12960 | 0.11 |
| `real_magnitude` | 0.9327 | [0.8173, 1.0000] | 0.3851 | 6868 | 6816 | 0.11 |
