# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `5` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `zrelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '4ASK', '16APSK']`. SNR (dB): `[-14, -10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary comparison. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4191 | 0.4098 | 0.0116 | 59406 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.3292 | 0.3287 | 0.0221 | 30151 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.3302 | 0.3299 | 0.0305 | 58777 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.3466 | 0.3465 | 0.0523 | 117639 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.4236 | 0.4220 | 0.0253 | 30599 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.2142 | 0.2203 | 0.0788 | 30151 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.4005 | 0.3991 | 0.0138 | 29703 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## Matched per-SNR test accuracy

| family | -14 dB | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.158 | 0.151 | 0.243 | 0.398 | 0.504 | 0.546 | 0.562 | 0.567 | 0.560 |
| `real_stacked` | 0.138 | 0.134 | 0.162 | 0.241 | 0.379 | 0.474 | 0.475 | 0.477 | 0.479 |
| `real_matched_params` | 0.144 | 0.132 | 0.152 | 0.255 | 0.405 | 0.454 | 0.476 | 0.480 | 0.470 |
| `real_matched_flops` | 0.146 | 0.142 | 0.160 | 0.269 | 0.416 | 0.492 | 0.509 | 0.498 | 0.486 |
| `real_polar` | 0.143 | 0.141 | 0.171 | 0.273 | 0.476 | 0.610 | 0.663 | 0.663 | 0.659 |
| `real_phase` | 0.132 | 0.144 | 0.149 | 0.179 | 0.265 | 0.286 | 0.270 | 0.275 | 0.284 |
| `real_magnitude` | 0.142 | 0.149 | 0.162 | 0.267 | 0.423 | 0.566 | 0.612 | 0.638 | 0.634 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4191 | 0.4098 | 0.0116 | 59406 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 1 | 0.4002 | 0.4016 | 0.0092 | 2167 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 1 | 0.4211 | 0.4238 | 0.0197 | 4239 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 1 | 0.4354 | 0.4355 | 0.0205 | 7911 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 12 | 0.4389 | 0.4427 | 0.0128 | 30599 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_phase` | 1 | 0.3516 | 0.3564 | 0.0065 | 2167 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 5 | 0.4005 | 0.3991 | 0.0138 | 29703 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
