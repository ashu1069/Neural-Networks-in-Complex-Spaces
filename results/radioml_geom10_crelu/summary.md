# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `5` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `crelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK', '4ASK', '16APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']`. SNR (dB): `[-14, -10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary comparison. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.3065 | 0.3055 | 0.0103 | 59796 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.2109 | 0.2090 | 0.0132 | 30346 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.1745 | 0.1746 | 0.0767 | 60343 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.1971 | 0.2029 | 0.0149 | 118026 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.2989 | 0.2967 | 0.0101 | 30794 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.1184 | 0.1198 | 0.0442 | 30346 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.2705 | 0.2683 | 0.0299 | 29898 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## Matched per-SNR test accuracy

| family | -14 dB | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.098 | 0.125 | 0.174 | 0.267 | 0.370 | 0.425 | 0.425 | 0.428 | 0.437 |
| `real_stacked` | 0.092 | 0.096 | 0.110 | 0.165 | 0.256 | 0.287 | 0.287 | 0.294 | 0.293 |
| `real_matched_params` | 0.100 | 0.103 | 0.108 | 0.132 | 0.210 | 0.228 | 0.233 | 0.231 | 0.227 |
| `real_matched_flops` | 0.105 | 0.101 | 0.110 | 0.153 | 0.256 | 0.275 | 0.273 | 0.274 | 0.279 |
| `real_polar` | 0.107 | 0.097 | 0.110 | 0.190 | 0.351 | 0.434 | 0.468 | 0.448 | 0.467 |
| `real_phase` | 0.100 | 0.100 | 0.100 | 0.112 | 0.117 | 0.135 | 0.139 | 0.138 | 0.137 |
| `real_magnitude` | 0.103 | 0.096 | 0.108 | 0.165 | 0.293 | 0.383 | 0.415 | 0.418 | 0.434 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.3065 | 0.3055 | 0.0103 | 59796 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 1 | 0.2900 | 0.2869 | 0.0082 | 2218 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 1 | 0.2930 | 0.2885 | 0.0205 | 4311 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 0 | 0.3036 | 0.2981 | 0.0092 | 30346 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_polar` | 1 | 0.3110 | 0.3115 | 0.0111 | 2330 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_phase` | 1 | 0.2465 | 0.2495 | 0.0074 | 2218 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 0 | 0.2799 | 0.2724 | 0.0047 | 7786 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
