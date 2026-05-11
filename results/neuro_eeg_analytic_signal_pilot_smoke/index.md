# EEG Analytic-Signal Pilot

## Plots

![accuracy heatmap](accuracy_heatmap.png)

![best accuracy by condition](best_accuracy_by_condition.png)

| condition | best | acc | complex | stacked | phase | polar | magnitude |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase_locking` | `real_phase` | 0.3500 | 0.2000 | 0.3000 | 0.3500 | 0.3000 | 0.2500 |
| `amplitude_event` | `real_polar` | 0.3000 | 0.2000 | 0.2500 | 0.2500 | 0.3000 | 0.3000 |
| `phase_amplitude_coupling` | `real_magnitude` | 0.2500 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2500 |
| `reference_phase_shift` | `real_phase` | 0.3000 | 0.2000 | 0.2000 | 0.3000 | 0.2500 | 0.2500 |
| `reference_phase_augmented` | `real_phase` | 0.5000 | 0.3500 | 0.4000 | 0.5000 | 0.4000 | 0.2500 |

Each condition directory contains `raw_runs.json`, `summary.json`, `summary.md`, `manifest.json`, and plots.
