# Synthetic RF Modulation Classification (Swept)

Selected configuration per model family from a random-search sweep of `16` trials x `3` seeds, following `docs/tuning_budget.md`. Stand-in for a future RadioML 2018.01A benchmark; numbers reflect the synthetic IQ + AWGN distribution, not the real dataset.

Architecture: `conv`. Activation (complex): `crelu`. Activation (real baselines): `relu`. Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[-10, -5, 0, 5, 10, 15, 20]`. Sample length: `128`.

| family | trial | val acc | test acc (mean) | test std | params | hyperparameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complex` | 1 | 0.8180 | 0.8217 | 0.0193 | 3974 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 1 | 0.7802 | 0.7708 | 0.0291 | 2099 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 0 | 0.7880 | 0.7918 | 0.0105 | 15033 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 1 | 0.7907 | 0.7940 | 0.0033 | 7779 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
