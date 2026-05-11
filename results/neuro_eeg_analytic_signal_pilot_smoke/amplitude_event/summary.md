# amplitude_event

Task: `amplitude_event`.

Question: Can models classify which sensor carries an amplitude burst when phase is independent of the label?

Expected signal: magnitude, polar, Cartesian, and complex views should learn; phase-only should remain near chance.

Preset: `smoke`. Seeds: `[0]`. Examples/class: `24`. Channels: `4`. Time steps: `32`. Train steps: `25`.

## Plots

![accuracy bar](accuracy_bar.png)

| family | acc | 95% CI | loss | params | madds | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.2000 | [0.2000, 0.2000] | 1.6146 | 2280 | 4480 | 0.01 |
| `real_stacked` | 0.2500 | [0.2500, 0.2500] | 1.7981 | 2164 | 2144 | 0.01 |
| `real_polar` | 0.3000 | [0.3000, 0.3000] | 1.5392 | 3188 | 3168 | 0.01 |
| `real_phase` | 0.2500 | [0.2500, 0.2500] | 2.1843 | 2164 | 2144 | 0.01 |
| `real_magnitude` | 0.3000 | [0.3000, 0.3000] | 1.2868 | 1140 | 1120 | 0.01 |
