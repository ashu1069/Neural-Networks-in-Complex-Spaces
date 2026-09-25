# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `6` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `crelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK']`. SNR (dB): `[-10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary comparison. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7194 | 0.7130 | 0.0274 | 58886 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.4874 | 0.5019 | 0.1341 | 29891 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.4065 | 0.4116 | 0.1212 | 58413 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.4122 | 0.4173 | 0.1302 | 117123 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.6653 | 0.6591 | 0.0492 | 30339 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.4195 | 0.4286 | 0.1499 | 29891 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.5286 | 0.5235 | 0.0498 | 29443 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## Matched per-SNR test accuracy

| family | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.358 | 0.373 | 0.582 | 0.808 | 0.872 | 0.905 | 0.903 | 0.904 |
| `real_stacked` | 0.353 | 0.346 | 0.385 | 0.556 | 0.583 | 0.593 | 0.600 | 0.599 |
| `real_matched_params` | 0.323 | 0.337 | 0.349 | 0.444 | 0.452 | 0.464 | 0.465 | 0.459 |
| `real_matched_flops` | 0.331 | 0.342 | 0.356 | 0.439 | 0.460 | 0.471 | 0.475 | 0.464 |
| `real_polar` | 0.333 | 0.329 | 0.403 | 0.719 | 0.859 | 0.871 | 0.874 | 0.885 |
| `real_phase` | 0.328 | 0.334 | 0.365 | 0.452 | 0.478 | 0.491 | 0.491 | 0.488 |
| `real_magnitude` | 0.343 | 0.315 | 0.359 | 0.488 | 0.634 | 0.668 | 0.691 | 0.690 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7194 | 0.7130 | 0.0274 | 58886 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 0 | 0.6687 | 0.6696 | 0.0299 | 7779 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 0 | 0.7114 | 0.7019 | 0.0092 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 0 | 0.7034 | 0.7010 | 0.0111 | 29891 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_polar` | 1 | 0.7130 | 0.7074 | 0.0147 | 2211 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_phase` | 1 | 0.6757 | 0.6733 | 0.0221 | 2099 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 1 | 0.5650 | 0.5533 | 0.0094 | 1987 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
