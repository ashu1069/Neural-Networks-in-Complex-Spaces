# Synthetic RF Modulation Classification

Stand-in for a future RadioML 2018.01A benchmark. Inputs are i.i.d. PSK/QAM symbols with AWGN at controlled SNR; no pulse shaping, no carrier offset, no fading. Numbers are not comparable to RadioML literature - they exist to compare baseline families on a sequence-shaped task.

Snapshot of one configuration. Numbers depend on platform BLAS and `git_commit`/`git_dirty` recorded in the manifest. Re-runs will not be byte-identical; see `docs/baselines.md` and `docs/tuning_budget.md` for the comparison rules.

Activation (complex): `crelu`. Activation (real baselines): `relu`. Seeds: `[0, 1, 2]`. Steps: `400`. Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[-10, -5, 0, 5, 10, 15, 20]`. Sample length: `128`.

| model | hidden | params | est. forward MAdds | accuracy mean | accuracy std | 95% CI | loss mean | train s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 64 | 25222 | 49920 | 0.4640 | 0.0427 | [0.4194, 0.5046] | 4.5 | 0.34 |
| `real_stacked` | 64 | 20803 | 20672 | 0.4783 | 0.0122 | [0.4679, 0.4918] | 4.79 | 0.16 |
| `real_matched_params` | 75 | 25203 | 25050 | 0.4802 | 0.0179 | [0.4652, 0.5000] | 4.6 | 0.16 |
| `real_matched_flops` | 129 | 50313 | 50052 | 0.4911 | 0.0060 | [0.4853, 0.4973] | 4.06 | 0.28 |

## Accuracy by SNR (dB)

| model | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 0.380 | 0.389 | 0.429 | 0.466 | 0.519 | 0.547 | 0.517 |
| `real_stacked` | 0.331 | 0.400 | 0.440 | 0.513 | 0.551 | 0.530 | 0.583 |
| `real_matched_params` | 0.346 | 0.355 | 0.434 | 0.524 | 0.524 | 0.577 | 0.603 |
| `real_matched_flops` | 0.344 | 0.385 | 0.425 | 0.543 | 0.564 | 0.579 | 0.598 |
