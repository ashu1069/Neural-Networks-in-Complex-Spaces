# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `6` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `crelu`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK']`. SNR (dB): `[-10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary comparison. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7210 | 0.7284 | 0.0314 | 58886 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.4976 | 0.5007 | 0.1304 | 29891 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.5070 | 0.5132 | 0.1397 | 58413 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.5414 | 0.5398 | 0.1041 | 117123 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.6896 | 0.6880 | 0.0278 | 30339 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.5005 | 0.5045 | 0.1382 | 29891 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.5560 | 0.5525 | 0.0308 | 29443 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## Matched per-SNR test accuracy

| family | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.338 | 0.387 | 0.605 | 0.835 | 0.901 | 0.928 | 0.919 | 0.915 |
| `real_stacked` | 0.342 | 0.350 | 0.390 | 0.554 | 0.584 | 0.591 | 0.598 | 0.595 |
| `real_matched_params` | 0.342 | 0.339 | 0.417 | 0.564 | 0.607 | 0.604 | 0.622 | 0.612 |
| `real_matched_flops` | 0.330 | 0.361 | 0.413 | 0.605 | 0.653 | 0.654 | 0.652 | 0.651 |
| `real_polar` | 0.343 | 0.311 | 0.444 | 0.720 | 0.899 | 0.913 | 0.946 | 0.928 |
| `real_phase` | 0.311 | 0.330 | 0.409 | 0.556 | 0.597 | 0.612 | 0.613 | 0.608 |
| `real_magnitude` | 0.334 | 0.318 | 0.365 | 0.467 | 0.644 | 0.759 | 0.759 | 0.774 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7210 | 0.7284 | 0.0314 | 58886 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 0 | 0.6713 | 0.6696 | 0.0301 | 7779 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 0 | 0.7074 | 0.6997 | 0.0141 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 1 | 0.7091 | 0.6993 | 0.0217 | 7779 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 1 | 0.7150 | 0.7097 | 0.0146 | 2211 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_phase` | 1 | 0.6789 | 0.6715 | 0.0268 | 2099 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 0 | 0.5603 | 0.5486 | 0.0083 | 7555 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
