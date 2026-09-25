# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `5` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `crelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '4ASK', '16APSK']`. SNR (dB): `[-14, -10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary comparison. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4442 | 0.4389 | 0.0123 | 59406 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_equivariant` | 5 | 0.4545 | 0.4477 | 0.0072 | 59399 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.3390 | 0.3369 | 0.0210 | 58777 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.3311 | 0.3321 | 0.0301 | 30151 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.4349 | 0.4327 | 0.0172 | 30599 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.3984 | 0.3952 | 0.0160 | 29703 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## Matched per-SNR test accuracy

| family | -14 dB | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.147 | 0.157 | 0.262 | 0.385 | 0.538 | 0.599 | 0.619 | 0.620 | 0.624 |
| `real_equivariant` | 0.134 | 0.142 | 0.175 | 0.349 | 0.553 | 0.663 | 0.680 | 0.665 | 0.669 |
| `real_matched_params` | 0.157 | 0.130 | 0.160 | 0.263 | 0.408 | 0.459 | 0.487 | 0.482 | 0.486 |
| `real_stacked` | 0.137 | 0.149 | 0.166 | 0.258 | 0.376 | 0.458 | 0.476 | 0.484 | 0.484 |
| `real_polar` | 0.148 | 0.153 | 0.165 | 0.279 | 0.493 | 0.635 | 0.670 | 0.678 | 0.674 |
| `real_magnitude` | 0.145 | 0.143 | 0.160 | 0.249 | 0.419 | 0.566 | 0.620 | 0.624 | 0.630 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4442 | 0.4389 | 0.0123 | 59406 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_equivariant` | 5 | 0.4545 | 0.4477 | 0.0072 | 59399 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 1 | 0.4201 | 0.4236 | 0.0205 | 4239 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 1 | 0.4002 | 0.4016 | 0.0092 | 2167 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 12 | 0.4389 | 0.4427 | 0.0128 | 30599 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_magnitude` | 0 | 0.4006 | 0.3941 | 0.0170 | 7687 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
