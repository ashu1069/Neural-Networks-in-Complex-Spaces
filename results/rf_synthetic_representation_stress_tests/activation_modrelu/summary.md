# activation_modrelu

Question: How much does complex activation `modrelu` matter?

Contradiction signal: the complex result changes enough across activations that the broad 'complex NN' claim is underspecified.

Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[0, 10, 20]`. Architecture: `conv`. Activation: `modrelu`. Train transform: `none`. Test transform: `none`.

| model | hidden | params | MAdds | accuracy | std | 95% CI | loss | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 16 | 2888 | 348352 | 0.7265 | 0.0700 | [0.6667, 0.8034] | 0.473 | 1.1 |
| `real_stacked` | 16 | 1523 | 92208 | 0.7279 | 0.0368 | [0.6880, 0.7607] | 0.436 | 0.33 |

## Accuracy by SNR (dB)

| model | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.641 | 0.756 | 0.782 |
| `real_stacked` | 0.624 | 0.769 | 0.791 |
