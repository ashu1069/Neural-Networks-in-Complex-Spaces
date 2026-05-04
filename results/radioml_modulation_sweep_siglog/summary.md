# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `6` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `siglog`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK']`. SNR (dB): `[-10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary paper table. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 3 | 0.7104 | 0.7014 | 0.0163 | 58886 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_stacked` | 3 | 0.4603 | 0.4689 | 0.1494 | 29891 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_params` | 3 | 0.4474 | 0.4657 | 0.1454 | 58413 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_flops` | 3 | 0.3977 | 0.4117 | 0.1226 | 117123 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |

## Matched per-SNR test accuracy

| family | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.325 | 0.371 | 0.519 | 0.792 | 0.885 | 0.900 | 0.923 | 0.897 |
| `real_stacked` | 0.323 | 0.345 | 0.379 | 0.516 | 0.541 | 0.560 | 0.537 | 0.550 |
| `real_matched_params` | 0.349 | 0.346 | 0.376 | 0.500 | 0.532 | 0.541 | 0.531 | 0.550 |
| `real_matched_flops` | 0.324 | 0.354 | 0.363 | 0.450 | 0.447 | 0.454 | 0.453 | 0.450 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 3 | 0.7104 | 0.7014 | 0.0163 | 58886 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_stacked` | 0 | 0.6680 | 0.6659 | 0.0280 | 7779 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 0 | 0.7097 | 0.6946 | 0.0114 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 0 | 0.7048 | 0.7023 | 0.0115 | 29891 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
