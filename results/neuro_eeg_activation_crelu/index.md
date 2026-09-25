# EEG Analytic-Signal Pilot

## Plots

![accuracy heatmap](accuracy_heatmap.png)

![best accuracy by condition](best_accuracy_by_condition.png)

| condition | best | acc | complex | stacked | phase | polar | magnitude |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `amplitude_event` | `real_magnitude` | 1.0000 | 0.8137 | 0.8225 | 0.2456 | 0.9936 | 1.0000 |
| `phase_amplitude_coupling` | `real_polar` | 1.0000 | 0.5779 | 0.8510 | 0.2475 | 1.0000 | 0.2500 |

Each condition directory contains `raw_runs.json`, `summary.json`, `summary.md`, `manifest.json`, and plots.
