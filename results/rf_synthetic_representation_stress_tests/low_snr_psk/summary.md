# low_snr_psk

Question: Does phase become too noisy in the low-SNR regime?

Contradiction signal: Cartesian or magnitude-heavy encodings beat phase/polar at low SNR, suggesting phase singularity/noise sensitivity.

Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[-10, -5, 0]`. Architecture: `conv`. Activation: `zrelu`. Train transform: `none`. Test transform: `none`.

| model | hidden | params | MAdds | accuracy | std | 95% CI | loss | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 16 | 2886 | 348352 | 0.5256 | 0.0513 | [0.4744, 0.5769] | 0.922 | 0.78 |
| `real_stacked` | 16 | 1523 | 92208 | 0.5256 | 0.0196 | [0.5043, 0.5427] | 0.914 | 0.31 |
| `real_matched_params` | 23 | 2993 | 184069 | 0.5370 | 0.0457 | [0.4872, 0.5769] | 0.887 | 0.32 |
| `real_matched_flops` | 32 | 5603 | 348256 | 0.5142 | 0.0437 | [0.4829, 0.5641] | 0.907 | 0.32 |
| `real_polar` | 16 | 1603 | 97328 | 0.4786 | 0.0186 | [0.4658, 0.5000] | 0.914 | 0.31 |
| `real_phase` | 16 | 1523 | 92208 | 0.4786 | 0.0186 | [0.4573, 0.4915] | 0.937 | 0.31 |
| `real_magnitude` | 16 | 1443 | 87088 | 0.3262 | 0.0243 | [0.2991, 0.3462] | 1.1 | 0.31 |

## Accuracy by SNR (dB)

| model | -10 dB | -5 dB | 0 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.427 | 0.453 | 0.697 |
| `real_stacked` | 0.368 | 0.538 | 0.671 |
| `real_matched_params` | 0.402 | 0.547 | 0.662 |
| `real_matched_flops` | 0.397 | 0.491 | 0.654 |
| `real_polar` | 0.359 | 0.444 | 0.632 |
| `real_phase` | 0.393 | 0.436 | 0.607 |
| `real_magnitude` | 0.312 | 0.333 | 0.333 |
