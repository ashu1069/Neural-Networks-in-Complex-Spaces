# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `6` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `zrelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK']`. SNR (dB): `[-10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary paper table. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.7417 | 0.7330 | 0.0154 | 15110 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_stacked` | 0 | 0.6684 | 0.6636 | 0.0298 | 7779 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 0 | 0.7118 | 0.7039 | 0.0075 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 0 | 0.7057 | 0.6990 | 0.0088 | 29891 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |

## Matched per-SNR test accuracy

| family | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.337 | 0.416 | 0.679 | 0.810 | 0.890 | 0.917 | 0.918 | 0.899 |
| `real_stacked` | 0.322 | 0.362 | 0.519 | 0.690 | 0.830 | 0.850 | 0.871 | 0.864 |
| `real_matched_params` | 0.337 | 0.377 | 0.513 | 0.754 | 0.884 | 0.909 | 0.935 | 0.923 |
| `real_matched_flops` | 0.333 | 0.350 | 0.518 | 0.749 | 0.887 | 0.915 | 0.925 | 0.915 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.7417 | 0.7330 | 0.0154 | 15110 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_stacked` | 0 | 0.6684 | 0.6636 | 0.0298 | 7779 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 0 | 0.7118 | 0.7039 | 0.0075 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 0 | 0.7057 | 0.6990 | 0.0088 | 29891 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
