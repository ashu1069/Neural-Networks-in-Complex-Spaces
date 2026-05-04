# RadioML 2018.01A Modulation Classification (Swept)

Random-search sweep of `16` trials x `6` seeds, following `docs/tuning_budget.md`. Real-data benchmark on the DeepSig RadioML 2018.01A archive (see `docs/radioml.md` for acquisition).

Architecture: `conv`. Activation (complex): `cardioid`. Activation (real baselines): `relu`. Modulations: `['BPSK', 'QPSK', '8PSK']`. SNR (dB): `[-10, -6, -2, 2, 6, 10, 14, 18]`. Sample length: `128`. Cap per class per SNR: `256`.

## Matched shared-trial comparison

Primary paper table. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7196 | 0.7282 | 0.0185 | 58886 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.4442 | 0.4570 | 0.1378 | 29891 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.4922 | 0.5028 | 0.1345 | 58413 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.4076 | 0.4159 | 0.1308 | 117123 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## Matched per-SNR test accuracy

| family | -10 dB | -6 dB | -2 dB | 2 dB | 6 dB | 10 dB | 14 dB | 18 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.356 | 0.446 | 0.640 | 0.792 | 0.885 | 0.890 | 0.916 | 0.903 |
| `real_stacked` | 0.331 | 0.335 | 0.381 | 0.506 | 0.515 | 0.531 | 0.517 | 0.538 |
| `real_matched_params` | 0.349 | 0.309 | 0.397 | 0.567 | 0.604 | 0.604 | 0.607 | 0.585 |
| `real_matched_flops` | 0.323 | 0.334 | 0.368 | 0.438 | 0.462 | 0.466 | 0.466 | 0.471 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7196 | 0.7282 | 0.0185 | 58886 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 0 | 0.6660 | 0.6615 | 0.0264 | 7779 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 0 | 0.7109 | 0.6950 | 0.0112 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 0 | 0.7029 | 0.7023 | 0.0100 | 29891 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
