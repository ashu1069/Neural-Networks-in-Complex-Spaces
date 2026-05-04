# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `6` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `crelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK']`. SNR (dB): `[-10, -5, 0, 5, 10, 15, 20]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary paper table. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7095 | 0.7217 | 0.0229 | 58886 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.4976 | 0.5112 | 0.1383 | 29891 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.4255 | 0.4239 | 0.1411 | 58413 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.4490 | 0.4458 | 0.1233 | 117123 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## Matched per-SNR test accuracy

| family | -10 dB | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: | ---: |
| `complex` | 0.337 | 0.718 | 0.917 | 0.916 |
| `real_stacked` | 0.355 | 0.495 | 0.595 | 0.600 |
| `real_matched_params` | 0.331 | 0.421 | 0.473 | 0.470 |
| `real_matched_flops` | 0.327 | 0.434 | 0.502 | 0.520 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7095 | 0.7217 | 0.0229 | 58886 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 0 | 0.6670 | 0.6803 | 0.0284 | 7779 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 0 | 0.6888 | 0.6968 | 0.0149 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 0 | 0.7000 | 0.6982 | 0.0097 | 29891 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
