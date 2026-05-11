# activation_crelu

Question: How much does complex activation `crelu` matter?

Contradiction signal: the complex result changes enough across activations that the broad 'complex NN' claim is underspecified.

Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[0, 10, 20]`. Architecture: `conv`. Activation: `crelu`. Train transform: `none`. Test transform: `none`.

## Plots

![accuracy bar](accuracy_bar.png)

![accuracy by snr](accuracy_by_snr.png)


| model | hidden | params | MAdds | accuracy | std | 95% CI | loss | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 16 | 2886 | 348352 | 0.7849 | 0.0696 | [0.7051, 0.8333] | 0.404 | 0.81 |
| `real_stacked` | 16 | 1523 | 92208 | 0.7279 | 0.0368 | [0.6880, 0.7607] | 0.436 | 0.33 |

## Accuracy by SNR (dB)

| model | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.654 | 0.825 | 0.876 |
| `real_stacked` | 0.624 | 0.769 | 0.791 |
