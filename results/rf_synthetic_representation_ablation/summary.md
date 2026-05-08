# Synthetic RF Modulation Classification

Stand-in for RadioML 2018.01A. Inputs are i.i.d. PSK/QAM symbols with AWGN at controlled SNR; no pulse shaping, no carrier offset, no fading. Numbers are not comparable to RadioML literature - they exist to compare baseline families on a sequence-shaped task.

Snapshot of one configuration. Numbers depend on platform BLAS and `git_commit`/`git_dirty` recorded in the manifest. Re-runs will not be byte-identical; see `docs/baselines.md` and `docs/tuning_budget.md` for the comparison rules.

Activation (complex): `zrelu`. Activation (real baselines): `relu`. Seeds: `[0, 1, 2]`. Steps: `160`. Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[0, 10, 20]`. Sample length: `64`.

| model | hidden | params | est. forward MAdds | accuracy mean | accuracy std | 95% CI | loss mean | train s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 16 | 2886 | 348352 | 0.8352 | 0.0335 | [0.8000, 0.8667] | 0.39 | 4.2 |
| `real_stacked` | 16 | 1523 | 92208 | 0.7704 | 0.0432 | [0.7222, 0.8056] | 0.427 | 1.4 |
| `real_polar` | 16 | 1603 | 97328 | 0.7259 | 0.0639 | [0.6611, 0.7889] | 0.473 | 1.4 |
| `real_phase` | 16 | 1523 | 92208 | 0.7722 | 0.0434 | [0.7222, 0.8000] | 0.437 | 1.4 |
| `real_magnitude` | 16 | 1443 | 87088 | 0.3333 | 0.0000 | [0.3333, 0.3333] | 1.1 | 1.4 |

## Accuracy by SNR (dB)

| model | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.678 | 0.878 | 0.950 |
| `real_stacked` | 0.711 | 0.800 | 0.800 |
| `real_polar` | 0.644 | 0.761 | 0.772 |
| `real_phase` | 0.694 | 0.806 | 0.817 |
| `real_magnitude` | 0.333 | 0.333 | 0.333 |
