# psk_representation

Question: Do phase-aware encodings explain PSK-family performance?

Contradiction signal: magnitude-only approaches phase/Cartesian accuracy, or real encodings consistently beat complex.

Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[0, 10, 20]`. Architecture: `conv`. Activation: `zrelu`. Train transform: `none`. Test transform: `none`.

| model | hidden | params | MAdds | accuracy | std | 95% CI | loss | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 16 | 2886 | 348352 | 0.8205 | 0.0128 | [0.8077, 0.8333] | 0.391 | 0.99 |
| `real_stacked` | 16 | 1523 | 92208 | 0.7279 | 0.0368 | [0.6880, 0.7607] | 0.436 | 0.34 |
| `real_matched_params` | 23 | 2993 | 184069 | 0.7764 | 0.0664 | [0.7222, 0.8504] | 0.416 | 0.35 |
| `real_matched_flops` | 32 | 5603 | 348256 | 0.7792 | 0.0580 | [0.7179, 0.8333] | 0.411 | 0.32 |
| `real_polar` | 16 | 1603 | 97328 | 0.7479 | 0.0421 | [0.7009, 0.7821] | 0.478 | 0.32 |
| `real_phase` | 16 | 1523 | 92208 | 0.7350 | 0.0446 | [0.6838, 0.7650] | 0.44 | 0.32 |
| `real_magnitude` | 16 | 1443 | 87088 | 0.3333 | 0.0000 | [0.3333, 0.3333] | 1.1 | 0.32 |

## Accuracy by SNR (dB)

| model | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.705 | 0.859 | 0.897 |
| `real_stacked` | 0.624 | 0.769 | 0.791 |
| `real_matched_params` | 0.667 | 0.825 | 0.838 |
| `real_matched_flops` | 0.645 | 0.829 | 0.863 |
| `real_polar` | 0.611 | 0.786 | 0.846 |
| `real_phase` | 0.585 | 0.786 | 0.833 |
| `real_magnitude` | 0.333 | 0.333 | 0.333 |
