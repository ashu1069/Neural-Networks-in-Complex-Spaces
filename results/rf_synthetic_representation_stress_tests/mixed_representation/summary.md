# mixed_representation

Question: Does the representation conclusion survive PSK+QAM together?

Contradiction signal: a hand-chosen real coordinate system matches or beats the complex model over mixed modulation families.

Modulations: `['bpsk', 'qpsk', '8psk', 'qam16', 'qam64']`. SNR (dB): `[0, 10, 20]`. Architecture: `conv`. Activation: `zrelu`. Train transform: `none`. Test transform: `none`.

| model | hidden | params | MAdds | accuracy | std | 95% CI | loss | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 16 | 2954 | 348480 | 0.5068 | 0.0231 | [0.4846, 0.5308] | 1.08 | 0.77 |
| `real_stacked` | 16 | 1557 | 92240 | 0.4812 | 0.0257 | [0.4564, 0.5077] | 1.09 | 0.32 |
| `real_matched_params` | 23 | 3041 | 184115 | 0.4795 | 0.0154 | [0.4641, 0.4949] | 1.08 | 0.33 |
| `real_matched_flops` | 32 | 5669 | 348320 | 0.4872 | 0.0268 | [0.4564, 0.5051] | 1.07 | 0.34 |
| `real_polar` | 16 | 1637 | 97360 | 0.4957 | 0.0681 | [0.4308, 0.5667] | 1.03 | 0.32 |
| `real_phase` | 16 | 1557 | 92240 | 0.4872 | 0.0315 | [0.4513, 0.5103] | 1.11 | 0.32 |
| `real_magnitude` | 16 | 1477 | 87120 | 0.3650 | 0.0065 | [0.3590, 0.3718] | 1.26 | 0.31 |

## Accuracy by SNR (dB)

| model | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.428 | 0.495 | 0.597 |
| `real_stacked` | 0.405 | 0.505 | 0.533 |
| `real_matched_params` | 0.403 | 0.503 | 0.533 |
| `real_matched_flops` | 0.421 | 0.497 | 0.544 |
| `real_polar` | 0.377 | 0.538 | 0.572 |
| `real_phase` | 0.392 | 0.495 | 0.574 |
| `real_magnitude` | 0.231 | 0.438 | 0.426 |
