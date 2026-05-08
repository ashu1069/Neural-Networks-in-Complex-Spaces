# fixed_rotation_psk

Question: Are models robust to an unseen global carrier phase offset?

Contradiction signal: all coordinate-dependent models fail under a fixed test rotation, weakening claims about native phase handling.

Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[0, 10, 20]`. Architecture: `conv`. Activation: `zrelu`. Train transform: `none`. Test transform: `fixed_rotation`.

| model | hidden | params | MAdds | accuracy | std | 95% CI | loss | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 16 | 2886 | 348352 | 0.2521 | 0.0299 | [0.2222, 0.2821] | 3.98 | 0.78 |
| `real_stacked` | 16 | 1523 | 92208 | 0.2464 | 0.0250 | [0.2179, 0.2650] | 3.44 | 0.32 |
| `real_polar` | 16 | 1603 | 97328 | 0.2721 | 0.0219 | [0.2479, 0.2906] | 2.81 | 0.32 |
| `real_phase` | 16 | 1523 | 92208 | 0.2621 | 0.0193 | [0.2436, 0.2821] | 3.14 | 0.32 |
| `real_magnitude` | 16 | 1443 | 87088 | 0.3276 | 0.0099 | [0.3162, 0.3333] | 1.1 | 0.32 |

## Accuracy by SNR (dB)

| model | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.248 | 0.252 | 0.256 |
| `real_stacked` | 0.308 | 0.265 | 0.167 |
| `real_polar` | 0.338 | 0.244 | 0.235 |
| `real_phase` | 0.350 | 0.252 | 0.184 |
| `real_magnitude` | 0.316 | 0.333 | 0.333 |
