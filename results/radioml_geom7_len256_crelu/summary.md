# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `5` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `crelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '4ASK', '16APSK']`. SNR (dB): `[-14, -10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `256`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary comparison. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4625 | 0.4660 | 0.0339 | 59406 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.3695 | 0.3706 | 0.0485 | 30151 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.3478 | 0.3460 | 0.0604 | 58777 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.3608 | 0.3618 | 0.0696 | 117639 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.4691 | 0.4728 | 0.0204 | 30599 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.2392 | 0.2404 | 0.0891 | 30151 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.4311 | 0.4332 | 0.0170 | 29703 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## Matched per-SNR test accuracy

| family | -14 dB | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.162 | 0.201 | 0.308 | 0.416 | 0.575 | 0.624 | 0.642 | 0.637 | 0.629 |
| `real_stacked` | 0.149 | 0.149 | 0.176 | 0.300 | 0.468 | 0.520 | 0.518 | 0.523 | 0.532 |
| `real_matched_params` | 0.144 | 0.142 | 0.159 | 0.270 | 0.425 | 0.490 | 0.501 | 0.484 | 0.501 |
| `real_matched_flops` | 0.141 | 0.139 | 0.167 | 0.253 | 0.451 | 0.526 | 0.521 | 0.530 | 0.530 |
| `real_polar` | 0.144 | 0.146 | 0.185 | 0.334 | 0.569 | 0.703 | 0.718 | 0.730 | 0.727 |
| `real_phase` | 0.150 | 0.151 | 0.158 | 0.189 | 0.260 | 0.304 | 0.315 | 0.319 | 0.317 |
| `real_magnitude` | 0.152 | 0.133 | 0.163 | 0.274 | 0.482 | 0.638 | 0.676 | 0.685 | 0.697 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4625 | 0.4660 | 0.0339 | 59406 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 0 | 0.4285 | 0.4299 | 0.0160 | 7911 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 1 | 0.4530 | 0.4557 | 0.0185 | 4239 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 1 | 0.4634 | 0.4713 | 0.0189 | 7911 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 12 | 0.4747 | 0.4753 | 0.0193 | 30599 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_phase` | 1 | 0.3765 | 0.3761 | 0.0139 | 2167 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 5 | 0.4311 | 0.4332 | 0.0170 | 29703 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
