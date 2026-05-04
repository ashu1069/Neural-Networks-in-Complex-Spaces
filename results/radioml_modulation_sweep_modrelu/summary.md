# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `6` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `modrelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK']`. SNR (dB): `[-10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary paper table. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 1 | 0.6634 | 0.6683 | 0.0310 | 3976 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 1 | 0.6331 | 0.6321 | 0.0348 | 2099 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 1 | 0.6963 | 0.6940 | 0.0622 | 3809 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 1 | 0.6813 | 0.6804 | 0.0379 | 7779 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |

## Matched per-SNR test accuracy

| family | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.357 | 0.408 | 0.639 | 0.699 | 0.779 | 0.817 | 0.833 | 0.814 |
| `real_stacked` | 0.348 | 0.340 | 0.504 | 0.701 | 0.762 | 0.794 | 0.804 | 0.803 |
| `real_matched_params` | 0.337 | 0.375 | 0.500 | 0.739 | 0.874 | 0.900 | 0.913 | 0.915 |
| `real_matched_flops` | 0.328 | 0.363 | 0.515 | 0.745 | 0.854 | 0.875 | 0.882 | 0.881 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 1 | 0.6634 | 0.6683 | 0.0310 | 3976 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 0 | 0.6677 | 0.6660 | 0.0262 | 7779 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 0 | 0.7145 | 0.7045 | 0.0096 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 0 | 0.7072 | 0.7010 | 0.0093 | 29891 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
