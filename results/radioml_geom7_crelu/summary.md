# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `5` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `crelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '4ASK', '16APSK']`. SNR (dB): `[-14, -10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary comparison. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4374 | 0.4369 | 0.0131 | 59406 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.3384 | 0.3363 | 0.0180 | 30151 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.3166 | 0.3197 | 0.0446 | 58777 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.3499 | 0.3576 | 0.0534 | 117639 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.4327 | 0.4298 | 0.0207 | 30599 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.2143 | 0.2176 | 0.0754 | 30151 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.4016 | 0.3980 | 0.0125 | 29703 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## Matched per-SNR test accuracy

| family | -14 dB | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.158 | 0.160 | 0.265 | 0.385 | 0.534 | 0.594 | 0.618 | 0.614 | 0.604 |
| `real_stacked` | 0.136 | 0.138 | 0.168 | 0.262 | 0.386 | 0.482 | 0.479 | 0.487 | 0.490 |
| `real_matched_params` | 0.146 | 0.137 | 0.177 | 0.263 | 0.376 | 0.431 | 0.449 | 0.452 | 0.447 |
| `real_matched_flops` | 0.149 | 0.138 | 0.161 | 0.276 | 0.426 | 0.504 | 0.526 | 0.521 | 0.517 |
| `real_polar` | 0.138 | 0.147 | 0.161 | 0.288 | 0.482 | 0.632 | 0.674 | 0.670 | 0.676 |
| `real_phase` | 0.137 | 0.141 | 0.149 | 0.172 | 0.260 | 0.278 | 0.271 | 0.273 | 0.276 |
| `real_magnitude` | 0.141 | 0.138 | 0.174 | 0.255 | 0.419 | 0.566 | 0.631 | 0.628 | 0.629 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4374 | 0.4369 | 0.0131 | 59406 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 1 | 0.4002 | 0.4016 | 0.0092 | 2167 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 1 | 0.4202 | 0.4233 | 0.0196 | 4239 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 1 | 0.4352 | 0.4352 | 0.0224 | 7911 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 12 | 0.4389 | 0.4427 | 0.0128 | 30599 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_phase` | 1 | 0.3516 | 0.3564 | 0.0065 | 2167 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 5 | 0.4016 | 0.3980 | 0.0125 | 29703 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
