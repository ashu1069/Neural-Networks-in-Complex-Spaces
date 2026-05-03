# Synthetic RF Modulation Classification (Swept)

Random-search sweep of `16` trials x `6` seeds, following `docs/tuning_budget.md`. Stand-in for a future RadioML 2018.01A benchmark; numbers reflect the synthetic IQ + AWGN distribution, not the real dataset.

Architecture: `conv`. Activation (complex): `crelu`. Activation (real baselines): `relu`. Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[-10, -5, 0, 5, 10, 15, 20]`. Sample length: `128`.

## Matched shared-trial comparison

Primary paper table. The trial index is selected by the complex family's mean validation accuracy, then every real baseline is reported at that same trial index so parameter/FLOP matching is with respect to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 1 | 0.8164 | 0.8191 | 0.0163 | 3974 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 1 | 0.7697 | 0.7740 | 0.0226 | 2099 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 1 | 0.7754 | 0.7769 | 0.0187 | 3809 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 1 | 0.7777 | 0.7810 | 0.0195 | 7779 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |

## Independent family winners

Diagnostic only. These rows show each family's own best validation trial, so their parameter counts are not guaranteed to be matched to the selected complex model.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 1 | 0.8164 | 0.8191 | 0.0163 | 3974 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 1 | 0.7697 | 0.7740 | 0.0226 | 2099 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 0 | 0.7872 | 0.7914 | 0.0127 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 0 | 0.7799 | 0.7865 | 0.0159 | 29891 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
